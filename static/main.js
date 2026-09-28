import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  cfg: null,
  selectedCandidateId: null,
  candidates: [],
  previewCanvas: null,     // offscreen canvas holding the current 8-bit preview (grayscale)
  exaggeration: 1,
  reliefMesh: null,
  reliefBase: null,       // base positions/uv of the curved face mesh (GLB path)
  shoulder: null,         // body mesh + rim links that let the shoulder follow the relief (see buildShoulder)
  gridNX: 400,
  gridNY: 340,
  debounceTimer: null,
  // Which part of the full source image maps onto the face -- mirrors
  // app/imaging.py's cover_fit_resize(zoom, offset_x, offset_y) exactly, so
  // the on-screen crop box previews precisely what the backend will crop.
  crop: { zoom: 1.0, offsetX: 0.0, offsetY: 0.0, naturalW: 0, naturalH: 0 },
};

// How the ring body follows the relief at the rim (GLB ring only).
const SHOULDER_FALLOFF_MM = 3.0; // how far down the shoulder the rim's lift fades out to nothing
const EDGE_SMOOTH_MM = 0.4;      // Gaussian sigma of the rim height along the edge (higher = broader waves)
const FACE_BLEND_MM = 0.6;       // band of the face next to the rim that eases into the smoothed edge

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------

let renderer, scene, camera, controls;
let ringGroup; // every ring mesh lives in here, so dragging can spin the ring like a turntable

// Drag sensitivity (radians per pixel): horizontal drag spins the ring,
// vertical drag tilts the camera up/down over it.
const DRAG_SPIN_PER_PX = 0.007;
const DRAG_TILT_PER_PX = 0.006;
// The camera starts turned this far (radians) to its right.
const CAMERA_YAW = 0.26;

async function initThree(cfg) {
  const canvas = document.getElementById("three-canvas");
  const viewer = document.getElementById("viewer");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xc9c6c3); // light gray like the studio backdrop, until the HDRI loads

  const pmremGenerator = new THREE.PMREMGenerator(renderer);

  const aspect = viewer.clientWidth / viewer.clientHeight;
  // Tight near/far (the whole scene spans ~20mm) to avoid depth-buffer
  // precision z-fighting at close/grazing viewing angles.
  camera = new THREE.PerspectiveCamera(35, aspect, 1, 600);
  // Frame the whole ring (band + face), not just the face, so the face's
  // small real-world scale relative to the band is visually obvious. The
  // band's loop is vertical (like a ring actually worn on a finger), so
  // the assembly is quite tall (roughly 2x the ring diameter, table to
  // bottom of the band) -- use a moderate front-ish elevation, like a
  // normal product photo, rather than a steep top-down angle.
  // The camera starts on the -Z side, turned CAMERA_YAW to its right;
  // dragging only tilts it (setupDragControls) while the ring spins.
  const ringHeight = cfg.ring_diameter_mm * 1.15 + cfg.ring_band_thickness_mm;
  const maxDim = Math.max(cfg.face_width_mm, cfg.face_height_mm, ringHeight);
  const targetY = -ringHeight * 0.45;
  const targetZ = cfg.ring_diameter_mm * 0.1;
  const start = new THREE.Spherical().setFromVector3(new THREE.Vector3(0, maxDim * 0.9, maxDim * 2.4));
  start.theta = Math.PI + CAMERA_YAW;
  camera.position.set(0, targetY, targetZ).add(new THREE.Vector3().setFromSpherical(start));
  camera.lookAt(0, targetY, targetZ);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, targetY, targetZ);
  controls.enableDamping = true;
  controls.enableRotate = false; // dragging is handled by setupDragControls
  controls.enablePan = false;
  controls.minDistance = maxDim * 0.5;
  controls.maxDistance = maxDim * 6;
  // Tilt all the way from above the ring to beneath it (to see its underside),
  // stopping just short of straight up/down where the view would flip.
  controls.minPolarAngle = 0.05;
  controls.maxPolarAngle = Math.PI - 0.05;
  setupDragControls();

  ringGroup = new THREE.Group();
  scene.add(ringGroup);

  // Gentle warm key light from above the camera's side, for a soft extra
  // highlight; kept low so it doesn't punch a hard pin-point into the metal.
  const keyLight = new THREE.DirectionalLight(0xfff4e6, 0.45);
  keyLight.position.set(-maxDim * 0.5, maxDim * 2, -maxDim);
  scene.add(keyLight);
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));

  await loadRing(cfg);
  // Not awaited: the UI is usable straight away, and the studio appears once
  // the HDRI has downloaded.
  loadStudio(pmremGenerator).catch((err) => console.warn("studio HDRI failed to load", err));

  onResize();
  window.addEventListener("resize", onResize);

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

/**
 * Horizontal drag spins the ring about its vertical axis (a turntable: the
 * camera and the studio stay put, so reflections sweep across the metal).
 * Vertical drag swings the camera up/down around the target, in the vertical
 * plane it is looking along. The mouse wheel / pinch still zooms via OrbitControls.
 */
function setupDragControls() {
  const el = renderer.domElement;
  let last = null;
  const sph = new THREE.Spherical();
  const offset = new THREE.Vector3();
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    last = { x: e.clientX, y: e.clientY };
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!last || !ringGroup) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    ringGroup.rotation.y += dx * DRAG_SPIN_PER_PX;
    offset.copy(camera.position).sub(controls.target);
    sph.setFromVector3(offset);
    sph.phi = clamp(sph.phi - dy * DRAG_TILT_PER_PX, controls.minPolarAngle, controls.maxPolarAngle);
    camera.position.copy(controls.target).add(offset.setFromSpherical(sph));
  });
  const end = () => { last = null; };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

function onResize() {
  const viewer = document.getElementById("viewer");
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

/**
 * 925 (sterling) silver, buffed to a near-mirror polish (like a jeweller's
 * render): bright neutral white metal, very low roughness so reflections are
 * sharp, no visible grain. ONE material instance is shared by every part of
 * the ring (body, face, relief) so color and texture match everywhere.
 * Raise `roughness` (0.15-0.3) for a satin/brushed feel instead.
 */
const SILVER = { color: 0xf3f2f0, metalness: 1.0, roughness: 0.05, envMapIntensity: 1.0 };
let silverMaterial = null;
function getSilverMaterial() {
  if (!silverMaterial) silverMaterial = new THREE.MeshStandardMaterial({ ...SILVER });
  return silverMaterial;
}

/**
 * "Oxidized" recesses on the face, like an antiqued signet ring: polished
 * silver has no shading of its own, so a relief under bright light washes
 * out. Jewellers blacken the grooves and polish the high points to make the
 * design readable; here the metal is darkened wherever the relief dips below
 * its surroundings (within `radiusMm`), so crevices beside raised detail go
 * dark while broad flat areas stay polished. `darkness` is how dark the
 * deepest crevices get (0 = off, 1 = black); `strength` how small a dip
 * already counts as fully dark.
 */
const PATINA = { radiusMm: 0.6, strength: 6.0, darkness: 0.85 };

// The face uses the same silver, plus vertex colours carrying the patina.
let faceMaterial = null;
function getFaceMaterial() {
  if (!faceMaterial) {
    faceMaterial = getSilverMaterial().clone();
    faceMaterial.vertexColors = true;
  }
  return faceMaterial;
}

/**
 * The ring's surroundings: a real photo-studio HDRI (Poly Haven "Monochrome
 * Studio 02", CC0; prepared by tools/prepare_skybox.py). It is both what the
 * silver reflects -- its strip softboxes and octabox give the crisp
 * highlights -- and the background, slightly blurred as if the camera were
 * focused on the ring, with the white seamless backdrop behind it.
 */
const STUDIO = {
  file: "/env/studio_env.hdr",
  backgroundBlur: 0.04,     // 0 = sharp; higher blurs the studio behind the ring more
  backgroundIntensity: 1.0,
  reflectionGain: 1.0,      // how strongly the metal reflects the studio
  // A soft lightbox above the ring that only the reflections see (the
  // studio's own ceiling is dark, so an upward-facing face would mirror
  // black). Size [w, h] and position [x, y, z] in mm relative to the ring's
  // centre; it faces the ring. It sits a little beyond the ring (+z) so the
  // face catches it from the starting camera angle too, not only from above.
  // `gradient` is the dim end's brightness relative to the bright end: a
  // graduated lightbox, so each slope of the relief mirrors a different
  // brightness and the design reads as shading (1 = evenly lit).
  topLight: { size: [100, 90], pos: [0, 50, 20], gain: 2.6, gradient: 0.08 },
  lightTint: [1.0, 0.98, 0.95], // soft warm white
  lightEdgeLevel: 0.5,          // how much the glow dims from its centre towards its edges
  lightEdgeFade: 0.45,          // outer fraction that fades smoothly to black, like a real softbox;
                                // a hard edge makes tiny surface wobbles show as jagged reflections
};
// Objects on this layer are seen by the reflection capture but never by the viewing camera.
const REFLECTION_LAYER = 1;

/**
 * A `w` x `h` light panel that glows from its centre and fades smoothly to
 * nothing at its edges, dimming along its height to `gradient` x at one end.
 */
function softLightPanel(w, h, gain, gradient = 1) {
  const L = STUDIO;
  const geo = new THREE.PlaneGeometry(w, h, 40, 40);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const u = (2 * pos.getX(i)) / w, v = (2 * pos.getY(i)) / h; // -1..1
    const glow = L.lightEdgeLevel + (1 - L.lightEdgeLevel) * Math.exp(-1.8 * (u * u + v * v) / 2);
    const fade = 1 - smoothstep((Math.max(Math.abs(u), Math.abs(v)) - (1 - L.lightEdgeFade)) / L.lightEdgeFade);
    const ramp = gradient + ((1 - gradient) * (v + 1)) / 2;
    const g = gain * glow * fade * ramp;
    colors.set([g * L.lightTint[0], g * L.lightTint[1], g * L.lightTint[2]], 3 * i);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
}

/**
 * Load the studio HDRI as the scene background, then capture the ring's
 * reflections: the studio plus the hidden top lightbox, seen from the ring's
 * position with the ring itself hidden. Everything captured is fixed in the
 * world, so spinning the ring needs no recapture. Throws if the HDRI can't load.
 */
async function loadStudio(pmremGenerator) {
  const hdr = await new RGBELoader().loadAsync(STUDIO.file);
  hdr.mapping = THREE.EquirectangularReflectionMapping;

  const center = new THREE.Box3().setFromObject(ringGroup).getCenter(new THREE.Vector3());
  const t = STUDIO.topLight;
  const top = softLightPanel(t.size[0], t.size[1], t.gain, t.gradient);
  top.position.set(center.x + t.pos[0], center.y + t.pos[1], center.z + t.pos[2]);
  top.lookAt(center);
  top.layers.set(REFLECTION_LAYER);
  scene.add(top);

  const cubeRT = new THREE.WebGLCubeRenderTarget(1024, { type: THREE.HalfFloatType });
  const cubeCam = new THREE.CubeCamera(0.5, 1000, cubeRT);
  cubeCam.children.forEach((c) => c.layers.enable(REFLECTION_LAYER));
  cubeCam.position.set(0, -4, 0); // just below the ring's face
  scene.background = hdr;
  scene.backgroundBlurriness = 0; // reflections see the studio sharp
  scene.backgroundIntensity = 1;
  ringGroup.visible = false;
  cubeCam.update(renderer, scene);
  ringGroup.visible = true;
  scene.backgroundBlurriness = STUDIO.backgroundBlur;
  scene.backgroundIntensity = STUDIO.backgroundIntensity;

  const oldEnv = scene.environment;
  scene.environment = pmremGenerator.fromCubemap(cubeRT.texture).texture;
  if (oldEnv) oldEnv.dispose();
  cubeRT.dispose();
  getSilverMaterial().envMapIntensity = STUDIO.reflectionGain;
  getFaceMaterial().envMapIntensity = STUDIO.reflectionGain;
}

function roundedRectShape(width, height, radius, bulge) {
  // A "cushion" outline: rounded corners plus a slight outward bulge along
  // each straight edge (via quadratic curves instead of straight lines).
  const w = width / 2, h = height / 2, r = Math.min(radius, w, h);
  const b = bulge || 0;
  const shape = new THREE.Shape();
  shape.moveTo(-w + r, -h);
  shape.quadraticCurveTo(0, -h - b, w - r, -h);
  shape.quadraticCurveTo(w, -h, w, -h + r);
  shape.quadraticCurveTo(w + b, 0, w, h - r);
  shape.quadraticCurveTo(w, h, w - r, h);
  shape.quadraticCurveTo(0, h + b, -w + r, h);
  shape.quadraticCurveTo(-w, h, -w, h - r);
  shape.quadraticCurveTo(-w - b, 0, -w, -h + r);
  shape.quadraticCurveTo(-w, -h, -w + r, -h);
  return shape;
}

function smoothstep(t) {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

/**
 * The whole shank (band + both shoulders) as ONE continuous swept mesh, so
 * there's no seam where a separate "shoulder" piece would otherwise meet
 * the band. The band's loop lies in a VERTICAL plane (XY, hole axis on Z)
 * like a real worn ring -- not lying flat -- so its top is tangent to the
 * table's underside and the rest of the loop curves downward and around,
 * in front of/below the table, matching a normal ring silhouette.
 *
 * Cross-section is circular around the bottom of the band (basis: radial
 * direction in the loop's own XY plane, and Z -- the loop's hole axis).
 * Near each end (approaching the table, at the TOP of the loop) it widens
 * and *twists* into a wide, flat, fluted fan (basis: Z for the fan's
 * spread along the table's side edge, and Y for its thin dimension,
 * matching the table's thickness) that meets the table's side edge.
 * Swept from the right table attachment point, down around the bottom of
 * the loop, up to the left table attachment point.
 */
function buildShankGeometry(params) {
  const {
    bandRadius, tubeRadius, bandPosY, bandPosZ, frontU, gapHalfAngle,
    halfW, slabThickness, tableInset,
    shoulderFraction, fluteCount, fluteDepth,
    segments, ringSegments,
  } = params;

  const u2 = frontU + gapHalfAngle; // one table attachment (sweep start)
  const u1 = frontU - gapHalfAngle + Math.PI * 2; // other table attachment (sweep end, unwrapped)
  const bandArc = u1 - u2;
  const shoulderArc = bandArc * shoulderFraction; // how much of the sweep each end's taper covers

  const positions = [];
  const center = new THREE.Vector3();
  const rawCenter = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const rightRadial = new THREE.Vector3();
  const upRadial = new THREE.Vector3(0, 0, 1); // loop's hole axis -- valid basis vector everywhere on the round band
  const rightFlat = new THREE.Vector3(0, 0, 1); // at the table: fan spreads along Z (the table's side edge)
  const upFlat = new THREE.Vector3(0, 1, 0); // at the table: thin axis matches the table's thickness (Y)

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const u = u2 + t * bandArc;

    // Distance (in swept arc-length terms) from whichever end is nearer.
    const distFromStart = u - u2;
    const distFromEnd = u1 - u;
    const nearDist = Math.min(distFromStart, distFromEnd);
    const s = 1 - smoothstep(nearDist / shoulderArc); // 0 = pure band, 1 = at the table
    const side = Math.sign(Math.cos(u)) || 1; // which table edge (+X/-X) this end blends toward

    rawCenter.set(bandRadius * Math.cos(u), bandRadius * Math.sin(u) + bandPosY, bandPosZ);
    // Only pull X (to the table's side) and Y (up to the table's underside)
    // toward a fixed attachment point -- Z stays on the band's own natural
    // path. Also blending Z toward an independently-chosen fixed point can
    // make the swept centerline fold back on itself (the raw circle's Z
    // there can be far from an arbitrary target), self-intersecting the
    // tube and producing thin spike artifacts.
    const targetX = side * (halfW - tableInset);
    const targetY = -slabThickness / 2;
    center.set(
      rawCenter.x + (targetX - rawCenter.x) * s,
      rawCenter.y + (targetY - rawCenter.y) * s,
      rawCenter.z
    );

    rightRadial.set(Math.cos(u), Math.sin(u), 0);
    right.copy(rightRadial).lerp(rightFlat, s).normalize();
    up.copy(upRadial).lerp(upFlat, s).normalize();

    // The shoulder widens into a broad, fairly flat fan (not a thin wire)
    // by the time it reaches the table, matching a real signet ring's
    // shoulder -- fluting depth stays modest relative to this width so
    // ridges read as gentle corrugation, not a jagged/forked silhouette.
    const endHalfWidth = 3.2, endHalfHeight = slabThickness / 2;
    const hw = tubeRadius + (endHalfWidth - tubeRadius) * s;
    const hh = tubeRadius + (endHalfHeight - tubeRadius) * s;
    const flute = fluteDepth * s; // flutes fade in only near the shoulders/table

    for (let j = 0; j <= ringSegments; j++) {
      const theta = (j / ringSegments) * Math.PI * 2;
      // Ripple only the width, not the already-thin height, to avoid
      // near-self-intersecting geometry on the thin axis.
      const rw = hw * (1 + flute * Math.cos(theta * fluteCount));
      const rh = hh;
      positions.push(
        center.x + right.x * rw * Math.cos(theta) + up.x * rh * Math.sin(theta),
        center.y + right.y * rw * Math.cos(theta) + up.y * rh * Math.sin(theta),
        center.z + right.z * rw * Math.cos(theta) + up.z * rh * Math.sin(theta)
      );
    }
  }

  const indices = [];
  const ringStride = ringSegments + 1;
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < ringSegments; j++) {
      const a = i * ringStride + j, b = a + 1;
      const c = (i + 1) * ringStride + j, d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * A stylized signet ring: a rounded-rectangle "cushion" table (unchanged
 * placement from the original viewer -- the relief mesh sits on it exactly
 * as before, at y=0) fused to a single continuous shank mesh (band +
 * fluted tapered shoulders, no internal seam). This is a proportional
 * visual preview only, not a manufacturing model of the shank -- the
 * exported heightmap/STL only ever describe the flat face; a manufacturer
 * determines actual shank/finger-size geometry separately.
 */
function buildRing(cfg) {
  const rim = 1.6;
  const slabThickness = 1.6;
  const cornerRadius = 2.0;
  const cushionBulge = 0.9;
  const halfW = cfg.face_width_mm / 2 + rim;
  const halfD = cfg.face_height_mm / 2 + rim;

  // DoubleSide: the extruded table and hand-built shank loft can have a
  // few faces with inconsistent winding after rotation -- double-siding
  // avoids backface-culling holes from unusual viewing angles. This is a
  // preview-only scene, so the minor cost is worth the robustness.
  const metalMat = getSilverMaterial();
  const bandMat = metalMat;

  // --- Table: rounded-rectangle "cushion" slab, top at y=0 (unchanged). ---
  const tableShape = roundedRectShape(halfW * 2, halfD * 2, cornerRadius, cushionBulge);
  const tableGeo = new THREE.ExtrudeGeometry(tableShape, { depth: slabThickness, bevelEnabled: false, curveSegments: 10 });
  tableGeo.rotateX(Math.PI / 2); // shape's XY -> world XZ; extrude depth -> world -Y (top stays at y=0)
  const table = new THREE.Mesh(tableGeo, metalMat);
  // Nudge the table a hair below y=0 so its top face isn't exactly
  // coplanar with the relief mesh (also at y=0) -- avoids z-fighting.
  table.position.y = -0.002;
  ringGroup.add(table);

  // --- Shank: one continuous mesh for the band + both fluted shoulders. ---
  // The band's loop lies in the vertical XY plane (hole axis on Z), like a
  // ring actually worn on a finger -- NOT flat/horizontal. u=PI/2 (top of
  // the loop, local point (0, R, 0)) is the point nearest the table.
  const bandRadius = cfg.ring_diameter_mm / 2;
  const tubeRadius = cfg.ring_band_thickness_mm / 2 + 0.4; // chunkier than a plain wire band
  const gapHalfAngle = Math.asin(Math.min(0.92, halfW / bandRadius));
  const FRONT_U = Math.PI / 2;
  const shankGeo = buildShankGeometry({
    bandRadius, tubeRadius,
    // Top of the loop (world y = bandRadius + bandPosY) lands at the
    // table's vertical center; bandPosZ centers the loop under the table
    // in depth, with a slight backward bias so the loop's front doesn't
    // poke out past the table's front edge.
    bandPosY: -slabThickness / 2 - bandRadius,
    bandPosZ: -bandRadius * 0.15,
    frontU: FRONT_U, gapHalfAngle,
    halfW, slabThickness, tableInset: 2.2,
    shoulderFraction: 0.4, fluteCount: 6, fluteDepth: 0.16,
    segments: 96, ringSegments: 28,
  });
  ringGroup.add(new THREE.Mesh(shankGeo, bandMat));
}

/**
 * Load the prepared ring model (tools/prepare_ring.py): a `body` mesh plus a
 * `face` mesh that covers the whole curved top of the head, in mm, hole axis
 * on Z, up = +Y. The relief displaces `face` vertices straight up by the
 * heightmap (see displaceFace), and the body's shoulder is bent up to meet
 * the displaced edge, fading out further down (see buildShoulder). Falls back
 * to the procedural stand-in ring (with a flat relief plane) if the model
 * can't be loaded.
 */
async function loadRing(cfg) {
  try {
    const [gltf, info] = await Promise.all([
      new GLTFLoader().loadAsync("/models/ring.glb"),
      fetch("/models/ring_face.json").then((r) => r.json()),
    ]);
    const silver = getSilverMaterial(); // same silver for body and face (the face adds its patina)
    let face = null, body = null;
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      o.material = silver;
      if (o.name === "face") face = o;
      if (o.name === "body") body = o;
    });
    if (!face) throw new Error("no 'face' mesh in ring.glb");
    if (!body) throw new Error("no 'body' mesh in ring.glb");
    ringGroup.add(gltf.scene);
    const g = face.geometry;
    face.material = getFaceMaterial();
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 3).fill(1), 3));
    state.reliefMesh = face;
    state.reliefBase = {
      pos: Float32Array.from(g.attributes.position.array),
      uv: Float32Array.from(g.attributes.uv.array),
    };
    state.shoulder = buildShoulder(body, state.reliefBase.pos, info.outline_xz_mm.length);
    setRingBadge("Ring model: ring.glb loaded");
  } catch (err) {
    console.warn("ring.glb failed to load; using procedural ring", err);
    setRingBadge("Ring model: FAILED to load ring.glb, showing fallback ring (" + (err && err.message || err) + ")", true);
    buildRing(cfg);
    buildReliefMesh(cfg);
  }
}

/**
 * Precompute how the body's shoulder follows the relief. Face vertices
 * 0..nRim-1 are the rim loop, at the same positions as the body's cut edge
 * (y = 0). Every body vertex within SHOULDER_FALLOFF_MM of the rim is linked
 * to its nearest rim point with a weight that is 1 at the rim and eases to 0
 * at the falloff distance; displaceFace lifts it by weight x that rim point's
 * (smoothed) relief height. The finger hole's inner surface is never moved.
 */
function buildShoulder(body, facePos, nRim) {
  const rimX = new Float32Array(nRim), rimZ = new Float32Array(nRim);
  for (let i = 0; i < nRim; i++) {
    rimX[i] = facePos[3 * i];
    rimZ[i] = facePos[3 * i + 2];
  }

  // Gaussian smoothing kernel along the loop, by arc length.
  const seg = new Float32Array(nRim); // seg[i] = length of edge i -> i+1
  for (let i = 0; i < nRim; i++) {
    const j = (i + 1) % nRim;
    seg[i] = Math.hypot(rimX[j] - rimX[i], rimZ[j] - rimZ[i]);
  }
  const kStart = [0], kIdx = [], kW = [];
  const reach = 3 * EDGE_SMOOTH_MM;
  for (let i = 0; i < nRim; i++) {
    const idx = [i], w = [1];
    for (const dir of [1, -1]) {
      let s = 0;
      for (let k = 1; k < nRim / 2; k++) {
        const j = (i + dir * k + nRim) % nRim;
        s += dir > 0 ? seg[(j - 1 + nRim) % nRim] : seg[j];
        if (s > reach) break;
        idx.push(j);
        w.push(Math.exp(-0.5 * (s / EDGE_SMOOTH_MM) ** 2));
      }
    }
    const total = w.reduce((a, b) => a + b, 0);
    for (let k = 0; k < idx.length; k++) {
      kIdx.push(idx[k]);
      kW.push(w[k] / total);
    }
    kStart.push(kIdx.length);
  }

  const nearestRim = (x, y, z) => {
    let best = 0, bestD2 = Infinity;
    for (let r = 0; r < nRim; r++) {
      const d2 = (x - rimX[r]) ** 2 + y * y + (z - rimZ[r]) ** 2;
      if (d2 < bestD2) { bestD2 = d2; best = r; }
    }
    return [best, Math.sqrt(bestD2)];
  };
  const insideOutline = (x, z) => {
    let inside = false;
    for (let i = 0, j = nRim - 1; i < nRim; j = i++) {
      if ((rimZ[i] > z) !== (rimZ[j] > z) &&
          x < ((rimX[j] - rimX[i]) * (z - rimZ[i])) / (rimZ[j] - rimZ[i]) + rimX[i]) inside = !inside;
    }
    return inside;
  };

  // Body vertices near the rim.
  const geo = body.geometry;
  const basePos = Float32Array.from(geo.attributes.position.array);
  const baseNrm = geo.attributes.normal.array;
  const bodyIdx = [], bodyRim = [], bodyW = [], seamBody = [], seamRim = [];
  for (let v = 0; v < basePos.length / 3; v++) {
    const x = basePos[3 * v], y = basePos[3 * v + 1], z = basePos[3 * v + 2];
    if (y < -SHOULDER_FALLOFF_MM) continue;
    // The finger hole's inner surface sits right under the face (inside the
    // outline, facing down): lifting it would put bumps inside the band.
    if (baseNrm[3 * v + 1] < -0.3 && insideOutline(x, z)) continue;
    const [r, d] = nearestRim(x, y, z);
    if (d >= SHOULDER_FALLOFF_MM) continue;
    bodyIdx.push(v);
    bodyRim.push(r);
    bodyW.push(0.5 * (1 + Math.cos((Math.PI * d) / SHOULDER_FALLOFF_MM)));
    if (d < 1e-3) { seamBody.push(v); seamRim.push(r); }
  }

  // Face vertices near the rim ease from their own height into the smoothed
  // rim height, so the face meets the shoulder without a step.
  const faceIdx = [], faceRim = [], faceT = [];
  for (let v = 0; v < facePos.length / 3; v++) {
    const [r, d] = nearestRim(facePos[3 * v], 0, facePos[3 * v + 2]);
    if (d >= FACE_BLEND_MM) continue;
    faceIdx.push(v);
    faceRim.push(r);
    faceT.push(smoothstep(d / FACE_BLEND_MM));
  }

  return {
    body, basePos, nRim,
    kStart: Int32Array.from(kStart), kIdx: Int32Array.from(kIdx), kW: Float32Array.from(kW),
    bodyIdx: Int32Array.from(bodyIdx), bodyRim: Int32Array.from(bodyRim), bodyW: Float32Array.from(bodyW),
    seamBody: Int32Array.from(seamBody), seamRim: Int32Array.from(seamRim),
    faceIdx: Int32Array.from(faceIdx), faceRim: Int32Array.from(faceRim), faceT: Float32Array.from(faceT),
  };
}

/** Small on-screen note saying which ring model is showing (helps debug stale caches). */
function setRingBadge(text, isError) {
  let el = document.getElementById("ringBadge");
  if (!el) {
    el = document.createElement("div");
    el.id = "ringBadge";
    el.style.cssText = "position:absolute;left:10px;bottom:8px;font:11px monospace;padding:3px 6px;border-radius:4px;background:#0008;pointer-events:none;";
    document.getElementById("viewer").appendChild(el);
  }
  el.textContent = text;
  el.style.color = isError ? "#ff8a80" : "#9ccc9c";
}

function buildReliefMesh(cfg) {
  const nx = state.gridNX;
  const ny = state.gridNY;
  const geo = new THREE.PlaneGeometry(cfg.face_width_mm, cfg.face_height_mm, nx - 1, ny - 1);
  geo.rotateX(-Math.PI / 2);

  const mat = getSilverMaterial();

  const mesh = new THREE.Mesh(geo, mat);
  ringGroup.add(mesh);
  state.reliefMesh = mesh;
}

/** Sample state.previewCanvas (grayscale 0..255) into an nx*ny Float32Array in 0..1. */
function sampleHeightGrid(nx, ny) {
  const off = document.createElement("canvas");
  off.width = nx;
  off.height = ny;
  const ctx = off.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(state.previewCanvas, 0, 0, nx, ny);
  const { data } = ctx.getImageData(0, 0, nx, ny);
  const out = new Float32Array(nx * ny);
  for (let i = 0; i < nx * ny; i++) {
    out[i] = data[i * 4] / 255; // grayscale, R channel
  }
  return out;
}

/** Bilinear sample of a w x h single-channel grid at uv (0..1). */
function sampleBilinear(grid, w, h, u, v) {
  const fx = Math.min(w - 1, Math.max(0, u * (w - 1)));
  const fy = Math.min(h - 1, Math.max(0, v * (h - 1)));
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const tx = fx - x0, ty = fy - y0;
  return (grid[y0 * w + x0] * (1 - tx) + grid[y0 * w + x1] * tx) * (1 - ty)
       + (grid[y1 * w + x0] * (1 - tx) + grid[y1 * w + x1] * tx) * ty;
}

/**
 * Blur a w x h single-channel grid: two passes of a separable box blur of
 * radius r (close to a Gaussian), with edges clamped. Done by hand rather than
 * with a canvas filter so it works in every browser.
 */
function boxBlur(src, w, h, r) {
  let a = Float32Array.from(src), b = new Float32Array(w * h);
  const pass = (from, to, len, count, stride, step) => {
    for (let line = 0; line < count; line++) {
      const base = line * stride;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += from[base + Math.min(len - 1, Math.max(0, k)) * step];
      for (let i = 0; i < len; i++) {
        to[base + i * step] = sum / (2 * r + 1);
        sum += from[base + Math.min(len - 1, i + r + 1) * step] - from[base + Math.max(0, i - r) * step];
      }
    }
  };
  for (let it = 0; it < 2; it++) {
    pass(a, b, w, h, w, 1); // rows
    pass(b, a, h, w, 1, w); // columns
  }
  return a;
}

/**
 * Displace the curved face mesh straight up (+Y) by the heightmap sampled at
 * each vertex's UV, and bend the body's shoulder up to meet it. UV u runs
 * along +X and v along +Z with image row 0 at -Z (see tools/prepare_ring.py).
 * Displacing vertically (not along the surface normal) keeps the relief from
 * pushing sideways over the steep rolled edge.
 *
 * The rim's heights are smoothed along the loop, so the edge rises in broad
 * waves instead of copying every fine spike of the texture; the face eases
 * into that smoothed edge over FACE_BLEND_MM, and the shoulder carries it
 * down, fading out over SHOULDER_FALLOFF_MM (see buildShoulder).
 */
function displaceFace() {
  const { pos: basePos, uv } = state.reliefBase;
  const src = state.previewCanvas;
  const w = src.width, h = src.height;
  const { data } = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h);
  const scaleMm = state.cfg.relief_max_mm * state.exaggeration;

  const geo = state.reliefMesh.geometry;
  const pos = geo.attributes.position;
  const n = pos.count;
  const raw = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) raw[i] = data[i * 4] / 255; // grayscale, R channel
  const blurred = boxBlur(raw, w, h, Math.max(1, Math.round((PATINA.radiusMm / state.cfg.face_width_mm) * w)));
  const heights = new Float32Array(n);
  const cavity = new Float32Array(n); // how far each point dips below its surroundings
  for (let i = 0; i < n; i++) {
    heights[i] = sampleBilinear(raw, w, h, uv[2 * i], uv[2 * i + 1]);
    cavity[i] = Math.max(0, sampleBilinear(blurred, w, h, uv[2 * i], uv[2 * i + 1]) - heights[i]);
  }
  const patinaFade = new Float32Array(n).fill(1); // no patina right at the rim, where the face meets the body

  const sh = state.shoulder;
  if (sh) {
    const edge = new Float32Array(sh.nRim);
    for (let r = 0; r < sh.nRim; r++) {
      let s = 0;
      for (let k = sh.kStart[r]; k < sh.kStart[r + 1]; k++) s += sh.kW[k] * heights[sh.kIdx[k]];
      edge[r] = s;
    }
    for (let k = 0; k < sh.faceIdx.length; k++) {
      const i = sh.faceIdx[k], e = edge[sh.faceRim[k]];
      heights[i] = e + (heights[i] - e) * sh.faceT[k];
      patinaFade[i] = sh.faceT[k];
    }

    const bp = sh.body.geometry.attributes.position;
    bp.array.set(sh.basePos);
    for (let k = 0; k < sh.bodyIdx.length; k++) {
      const v = sh.bodyIdx[k];
      bp.array[3 * v + 1] += sh.bodyW[k] * edge[sh.bodyRim[k]] * scaleMm;
    }
    bp.needsUpdate = true;
    sh.body.geometry.computeVertexNormals();
    sh.body.geometry.computeBoundingSphere();
  }

  for (let i = 0; i < n; i++) {
    pos.setXYZ(i, basePos[3 * i], basePos[3 * i + 1] + heights[i] * scaleMm, basePos[3 * i + 2]);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  const col = geo.attributes.color;
  if (col) {
    for (let i = 0; i < n; i++) {
      const c = 1 - PATINA.darkness * smoothstep(cavity[i] * PATINA.strength) * patinaFade[i];
      col.setXYZ(i, c, c, c);
    }
    col.needsUpdate = true;
  }

  if (sh) {
    // Face and body each see only their own triangles at the shared rim;
    // average the two so the edge shades as one rounded surface, not a crease.
    const fn = geo.attributes.normal, bn = sh.body.geometry.attributes.normal;
    const acc = new Float32Array(sh.nRim * 3);
    for (let r = 0; r < sh.nRim; r++) {
      acc[3 * r] = fn.getX(r); acc[3 * r + 1] = fn.getY(r); acc[3 * r + 2] = fn.getZ(r);
    }
    for (let k = 0; k < sh.seamBody.length; k++) {
      const b = sh.seamBody[k], r = sh.seamRim[k];
      acc[3 * r] += bn.getX(b); acc[3 * r + 1] += bn.getY(b); acc[3 * r + 2] += bn.getZ(b);
    }
    const nv = new THREE.Vector3();
    for (let r = 0; r < sh.nRim; r++) {
      nv.set(acc[3 * r], acc[3 * r + 1], acc[3 * r + 2]).normalize();
      fn.setXYZ(r, nv.x, nv.y, nv.z);
    }
    for (let k = 0; k < sh.seamBody.length; k++) {
      const r = sh.seamRim[k];
      bn.setXYZ(sh.seamBody[k], fn.getX(r), fn.getY(r), fn.getZ(r));
    }
    fn.needsUpdate = true;
    bn.needsUpdate = true;
  }
}

function updateMeshHeights() {
  if (!state.previewCanvas || !state.reliefMesh) return;
  if (state.reliefBase) {
    displaceFace();
    return;
  }
  const nx = state.gridNX;
  const ny = state.gridNY;
  const heights = sampleHeightGrid(nx, ny);
  const reliefMaxMm = state.cfg.relief_max_mm;
  const exaggeration = state.exaggeration;

  const posAttr = state.reliefMesh.geometry.attributes.position;
  // PlaneGeometry vertex order: row-major, y=0 at top row (before rotation).
  // After rotateX(-PI/2): plane's local Y (row index) maps to -Z (so image
  // row 0 (top of image) ends up at -Z, matching a normal top-down image).
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const vertIndex = row * nx + col;
      const h = heights[row * nx + col] * reliefMaxMm * exaggeration;
      posAttr.setY(vertIndex, h);
    }
  }
  posAttr.needsUpdate = true;
  state.reliefMesh.geometry.computeVertexNormals();
  state.reliefMesh.geometry.computeBoundingSphere();
}

// ---------------------------------------------------------------------------
// Hard-coded test heightmap (Milestone 1 — shown until a real design loads)
// ---------------------------------------------------------------------------

function makeTestHeightmapCanvas(cfg) {
  const w = 350, h = 300;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);

  // A few concentric rings + a star, purely to sanity-check the displacement
  // pipeline (grid sampling, normals, exaggeration) before any AI/backend flow.
  const cx = w / 2, cy = h / 2;
  const rings = [1.0, 0.75, 0.5, 0.25];
  const maxR = Math.min(w, h) * 0.38;
  rings.forEach((f, i) => {
    const v = Math.round(255 * ((i + 1) / rings.length));
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * f, 0, Math.PI * 2);
    ctx.fill();
  });

  // Keep a black margin so it resembles a real edge-masked heightmap.
  const marginPx = (cfg.edge_margin_mm / cfg.face_width_mm) * w;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, marginPx);
  ctx.fillRect(0, h - marginPx, w, marginPx);
  ctx.fillRect(0, 0, marginPx, h);
  ctx.fillRect(w - marginPx, 0, marginPx, h);

  return canvas;
}

// ---------------------------------------------------------------------------
// Backend API
// ---------------------------------------------------------------------------

async function fetchConfig() {
  const res = await fetch("/api/config");
  return res.json();
}

async function generateCandidates(prompt, preset) {
  // Backend always generates exactly one image per call (each call is a
  // real, billed API request) -- click "Generate" again to add another.
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, preset }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadPhoto(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function processCandidate(candidateId, params) {
  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_id: candidateId, ...params }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function loadImageToCanvas(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      resolve(canvas);
    };
    img.onerror = reject;
    img.src = url;
  });
}

// ---------------------------------------------------------------------------
// Crop / pan / zoom -- mirrors app/imaging.py's cover_fit_resize math
// ---------------------------------------------------------------------------

/** Same math as cover_fit_resize, in normalized (0..1) image-fraction units. */
function computeCropGeometry(naturalW, naturalH, faceW, faceH, zoom) {
  const targetAspect = faceW / faceH;
  const srcAspect = naturalW / naturalH;
  let baseWNorm, baseHNorm;
  if (srcAspect > targetAspect) {
    baseHNorm = 1.0;
    baseWNorm = targetAspect / srcAspect;
  } else {
    baseWNorm = 1.0;
    baseHNorm = srcAspect / targetAspect;
  }
  const z = Math.max(1.0, zoom);
  const cropWNorm = baseWNorm / z;
  const cropHNorm = baseHNorm / z;
  const maxOffsetXNorm = Math.max(0, (1 - cropWNorm) / 2);
  const maxOffsetYNorm = Math.max(0, (1 - cropHNorm) / 2);
  return { cropWNorm, cropHNorm, maxOffsetXNorm, maxOffsetYNorm };
}

function renderCropBox() {
  const { naturalW, naturalH } = state.crop;
  if (!naturalW || !naturalH || !state.cfg) return;
  const { cropWNorm, cropHNorm, maxOffsetXNorm, maxOffsetYNorm } = computeCropGeometry(
    naturalW, naturalH, state.cfg.face_width_mm, state.cfg.face_height_mm, state.crop.zoom
  );
  const centerXNorm = 0.5 + state.crop.offsetX * maxOffsetXNorm;
  const centerYNorm = 0.5 + state.crop.offsetY * maxOffsetYNorm;
  const box = document.getElementById("cropperBox");
  box.style.left = `${(centerXNorm - cropWNorm / 2) * 100}%`;
  box.style.top = `${(centerYNorm - cropHNorm / 2) * 100}%`;
  box.style.width = `${cropWNorm * 100}%`;
  box.style.height = `${cropHNorm * 100}%`;
}

function loadCropperImage(designId) {
  return new Promise((resolve, reject) => {
    const img = document.getElementById("cropperImage");
    const container = document.getElementById("cropperContainer");
    img.onload = () => {
      state.crop.naturalW = img.naturalWidth;
      state.crop.naturalH = img.naturalHeight;
      state.crop.zoom = 1.0;
      state.crop.offsetX = 0.0;
      state.crop.offsetY = 0.0;
      // Match the container's aspect ratio to the image so container
      // fractions map 1:1 to image-normalized fractions (clamped so a
      // very extreme photo aspect can't blow up the panel layout).
      const aspect = clamp(img.naturalWidth / img.naturalHeight, 0.4, 2.5);
      container.style.aspectRatio = `${aspect}`;
      document.getElementById("cropZoom").value = "1";
      document.getElementById("cropZoomVal").textContent = "1.00x";
      renderCropBox();
      resolve();
    };
    img.onerror = reject;
    img.src = `/api/designs/${designId}/full.png?t=${Date.now()}`;
  });
}

function wireCropDrag() {
  const box = document.getElementById("cropperBox");
  const container = document.getElementById("cropperContainer");
  let dragging = false;
  let startClientX = 0, startClientY = 0;
  let startOffsetX = 0, startOffsetY = 0;

  box.addEventListener("pointerdown", (e) => {
    dragging = true;
    box.setPointerCapture(e.pointerId);
    startClientX = e.clientX;
    startClientY = e.clientY;
    startOffsetX = state.crop.offsetX;
    startOffsetY = state.crop.offsetY;
  });

  box.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const dxNorm = (e.clientX - startClientX) / rect.width;
    const dyNorm = (e.clientY - startClientY) / rect.height;
    const { maxOffsetXNorm, maxOffsetYNorm } = computeCropGeometry(
      state.crop.naturalW, state.crop.naturalH,
      state.cfg.face_width_mm, state.cfg.face_height_mm, state.crop.zoom
    );
    // Below this there's negligible real room to pan (can happen at zoom=1
    // when the source is already very close to the face aspect ratio) --
    // dividing by a near-zero max would amplify a tiny mouse move into an
    // instant snap to the clamped edge, which feels like a broken drag.
    const MIN_PANNABLE_NORM = 0.01;
    state.crop.offsetX = clamp(
      startOffsetX + (maxOffsetXNorm > MIN_PANNABLE_NORM ? dxNorm / maxOffsetXNorm : 0), -1, 1
    );
    state.crop.offsetY = clamp(
      startOffsetY + (maxOffsetYNorm > MIN_PANNABLE_NORM ? dyNorm / maxOffsetYNorm : 0), -1, 1
    );
    renderCropBox();
  });

  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try { box.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    debouncedRefresh();
  };
  box.addEventListener("pointerup", endDrag);
  box.addEventListener("pointercancel", endDrag);
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

function currentParams() {
  return {
    invert: document.getElementById("invert").checked,
    flip_h: document.getElementById("flipH").checked,
    flip_v: document.getElementById("flipV").checked,
    gamma: 1.0,
    contrast: 1.0,
    blur_mm: parseFloat(document.getElementById("blur").value),
    levels: parseInt(document.getElementById("levels").value, 10),
    min_feature_mm: parseFloat(document.getElementById("minFeature").value),
    relief_height_mm: parseFloat(document.getElementById("reliefHeight").value),
    crop_zoom: state.crop.zoom,
    crop_offset_x: state.crop.offsetX,
    crop_offset_y: state.crop.offsetY,
  };
}

function setStatus(elId, msg, isError = false) {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function renderReport(report) {
  const box = document.getElementById("reportBox");
  const passClass = report.passed ? "good" : "bad";
  box.innerHTML = `
    <div>Coverage: ${report.coverage_percent.toFixed(1)}%</div>
    <div>Relief volume: ${report.relief_volume_mm3.toFixed(2)} mm&sup3;</div>
    <div>Estimated weight (silver, relief only): ${report.estimated_weight_g.toFixed(3)} g</div>
    <div class="${passClass}">${report.passed ? "Looks manufacturable" : "Needs attention"}</div>
  `;
  const warningsBox = document.getElementById("warningsBox");
  warningsBox.innerHTML = "";
  for (const w of report.warnings || []) {
    const div = document.createElement("div");
    div.className = "warning";
    div.textContent = w;
    warningsBox.appendChild(div);
  }
  document.getElementById("reportSection").hidden = false;
}

async function refreshFromBackend() {
  if (!state.selectedCandidateId) return;
  setStatus("processStatus", "Processing...");
  try {
    const result = await processCandidate(state.selectedCandidateId, currentParams());
    const canvas = await loadImageToCanvas(result.preview_url + `?t=${Date.now()}`);
    state.previewCanvas = canvas;
    updateMeshHeights();
    renderReport(result.report);
    document.getElementById("downloadHeightmap").href = result.heightmap_url;
    document.getElementById("downloadParams").href = result.params_url;
    document.getElementById("downloadStl").href = `/api/designs/${state.selectedCandidateId}/model.stl`;
    setStatus("processStatus", "");
  } catch (err) {
    console.error(err);
    setStatus("processStatus", "Processing failed: " + err.message, true);
  }
}

function debouncedRefresh() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(refreshFromBackend, 300);
}

function wireSliderDisplay(id, suffix = "") {
  const input = document.getElementById(id);
  const label = document.getElementById(id + "Val");
  const update = () => (label.textContent = input.value + suffix);
  input.addEventListener("input", update);
  update();
}

function populatePresets(cfg) {
  const select = document.getElementById("preset");
  select.innerHTML = "";
  for (const [id, p] of Object.entries(cfg.presets)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = p.label;
    if (id === cfg.default_preset) opt.selected = true;
    select.appendChild(opt);
  }
}

function applyPresetDefaults(cfg) {
  const presetId = document.getElementById("preset").value;
  const p = cfg.presets[presetId];
  if (!p) return;
  document.getElementById("levels").value = p.levels;
  document.getElementById("blur").value = p.blur_mm;
  document.getElementById("levelsVal").textContent = p.levels;
  document.getElementById("blurVal").textContent = p.blur_mm;
}

function renderGallery(candidates) {
  const gallery = document.getElementById("gallery");
  gallery.innerHTML = "";
  for (const c of candidates) {
    const img = document.createElement("img");
    img.src = c.thumbnail_url;
    img.dataset.candidateId = c.candidate_id;
    img.addEventListener("click", () => selectCandidate(c.candidate_id, img));
    gallery.appendChild(img);
  }
}

async function selectCandidate(candidateId, imgEl) {
  state.selectedCandidateId = candidateId;
  document.querySelectorAll("#gallery img").forEach((el) => el.classList.remove("selected"));
  if (imgEl) imgEl.classList.add("selected");
  document.getElementById("cropSection").hidden = false;
  document.getElementById("paramsSection").hidden = false;
  applyPresetDefaults(state.cfg);
  await loadCropperImage(candidateId);
  refreshFromBackend();
}

function wireUI(cfg) {
  populatePresets(cfg);
  wireCropDrag();

  document.getElementById("cropZoom").addEventListener("input", (e) => {
    state.crop.zoom = parseFloat(e.target.value);
    document.getElementById("cropZoomVal").textContent = state.crop.zoom.toFixed(2) + "x";
    renderCropBox();
  });
  document.getElementById("cropZoom").addEventListener("change", debouncedRefresh);

  ["reliefHeight", "levels", "blur", "minFeature"].forEach((id) => {
    wireSliderDisplay(id);
    document.getElementById(id).addEventListener("change", debouncedRefresh);
  });
  document.getElementById("invert").addEventListener("change", debouncedRefresh);
  document.getElementById("flipH").addEventListener("change", debouncedRefresh);
  document.getElementById("flipV").addEventListener("change", debouncedRefresh);

  wireSliderDisplay("exaggeration", "x");
  document.getElementById("exaggeration").addEventListener("input", (e) => {
    state.exaggeration = parseFloat(e.target.value);
    updateMeshHeights();
  });

  document.getElementById("photoUploadInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus("generateStatus", "Uploading photo...");
    try {
      const candidate = await uploadPhoto(file);
      state.candidates = [...state.candidates, candidate];
      renderGallery(state.candidates);
      const newImg = document.querySelector(`#gallery img[data-candidate-id="${candidate.candidate_id}"]`);
      selectCandidate(candidate.candidate_id, newImg);
      setStatus("generateStatus", "Photo uploaded and selected below.");
    } catch (err) {
      console.error(err);
      setStatus("generateStatus", "Upload failed: " + err.message, true);
    } finally {
      e.target.value = "";
    }
  });

  document.getElementById("generateBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value.trim();
    if (!prompt) {
      setStatus("generateStatus", "Enter a description first.", true);
      return;
    }
    const preset = document.getElementById("preset").value;
    const btn = document.getElementById("generateBtn");
    btn.disabled = true;
    setStatus("generateStatus", "Generating 1 image...");
    try {
      const result = await generateCandidates(prompt, preset);
      // Each click generates exactly one image (one billed API call) and
      // adds it to the gallery -- click again to add more for comparison.
      state.candidates = [...state.candidates, ...result.candidates];
      renderGallery(state.candidates);
      setStatus("generateStatus", "Image added below. Click Generate again for another, or pick one to preview.");
      if (result.warnings && result.warnings.length) {
        setStatus("generateStatus", result.warnings.join(" "));
      }
      if (result.candidates.length) {
        const newCandidate = result.candidates[0];
        const newImg = document.querySelector(`#gallery img[data-candidate-id="${newCandidate.candidate_id}"]`);
        selectCandidate(newCandidate.candidate_id, newImg);
      }
    } catch (err) {
      console.error(err);
      setStatus("generateStatus", "Generation failed: " + err.message, true);
    } finally {
      btn.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function main() {
  const cfg = await fetchConfig();
  state.cfg = cfg;

  await initThree(cfg);

  state.previewCanvas = makeTestHeightmapCanvas(cfg);
  updateMeshHeights();

  wireUI(cfg);
}

main();
