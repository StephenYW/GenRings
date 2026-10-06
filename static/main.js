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
  exaggerateUp: false,    // preview exaggeration grows peaks upward from the base instead of sinking the design downward
  faceTiltDeg: 8,         // design area edge: the relief ring's top that tilts less than this (set from relief.json)
  ridgeShiftMm: 0.3,
  ringVolumeMm3: null,    // the ring on show, with the design at its true height (for the total weight)      // recess rings: how far the ridge's inner wall is slid outward (the "Ridge wall" slider)
  reliefMesh: null,       // the ring mesh that carries the relief (only the relief ring, see RINGS)
  reliefBase: null,       // its undisplaced positions/normals, uv, relief weights and region size
  debounceTimer: null,
  // Which part of the full source image maps onto the face -- mirrors
  // app/imaging.py's cover_fit_resize(zoom, offset_x, offset_y) exactly, so
  // the on-screen crop box previews precisely what the backend will crop.
  crop: { zoom: 1.0, offsetX: 0.0, offsetY: 0.0, naturalW: 0, naturalH: 0 },
};

/**
 * The ring library (tools/prepare_rings.py): every shape in UK sizes H-Z as
 * static/rings/<Shape>/<Size>.glb, listed in catalog.json. One ring (S
 * Square for now, see relief.json) carries the relief on its flat top: the
 * part that tilts less than an angle set live with the "Design area" slider.
 * Its first zone_vertex_count vertices (and first zone_face_count faces) are
 * the top faces any angle up to the table's maximum could include.
 * computeDesignArea works out the design area for the angle; displaceFace
 * applies the design, which ends at the area's edge with a hard edge.
 */
const RINGS = {
  catalog: null,
  reliefs: {},         // relief.json's rings: "Shape/Size" -> that design ring's design-area info
  relief: null,        // the info of the design ring the design currently goes on
  reliefKey: null,     // ...and its "Shape/Size" (sent with each heightmap request)
  tiltInfo: null,      // the "tilt" design ring's info (drives the Design area slider)
  current: null,       // { shape, size } on show
  loadToken: 0,        // ignores a slow load that a newer selection has overtaken
};

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
  // (frameRing re-aims it at each ring once loaded.)
  const maxDim = 26; // mm: a mid-size ring, until the first one loads
  const start = new THREE.Spherical().setFromVector3(new THREE.Vector3(0, maxDim * 0.9, maxDim * 2.4));
  start.theta = Math.PI + CAMERA_YAW;
  camera.position.set(0, 0, 0).add(new THREE.Vector3().setFromSpherical(start));
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
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

  await loadRingLibrary();
  // Not awaited: the UI is usable straight away, and the studio appears once
  // the HDRI has downloaded. (It centres its reflection capture on the first
  // ring, so it starts after that has loaded.)
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
  topLight: { size: [100, 90], pos: [0, 60, 20], gain: 2.6, gradient: 0.08 },
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
  cubeCam.position.copy(center); // from the ring's centre (the rings are all about the same size)
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

function smoothstep(t) {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

/** Load the catalog and relief info, build the ring menu, and show the default ring. */
async function loadRingLibrary() {
  const [catalog, relief] = await Promise.all([
    fetch("/rings/catalog.json", { cache: "no-store" }).then((r) => r.json()),
    fetch("/rings/relief.json", { cache: "no-store" }).then((r) => r.json()).catch(() => null),
  ]);
  RINGS.catalog = catalog;
  RINGS.reliefs = (relief && relief.rings) || {};
  RINGS.reliefVersion = relief && relief.version;
  RINGS.reliefKey = relief && relief.default;
  RINGS.relief = RINGS.reliefs[RINGS.reliefKey] || null;
  RINGS.tiltInfo = Object.values(RINGS.reliefs).find((r) => r.mode === "tilt") || null;
  const tiltInfo = RINGS.tiltInfo;
  if (tiltInfo) {
    const t = tiltInfo.tilt_table;
    const slider = document.getElementById("designTilt");
    slider.min = t[0].deg;
    slider.max = t[t.length - 1].deg;
    slider.step = tiltInfo.tilt_step_deg;
    slider.value = tiltInfo.default_tilt_deg;
    setDesignTilt(tiltInfo.default_tilt_deg, false);
  }
  buildRingMenu();
  const d = catalog.default;
  await selectRing(d.shape, d.size);
}

function shapeEntry(id) {
  return RINGS.catalog.shapes.find((s) => s.id === id);
}

function isReliefRing(shape, size) {
  return Boolean(RINGS.reliefs[`${shape}/${size}`]);
}

/**
 * Show one ring from the library: load its GLB, give it the silver finish,
 * and frame the camera on it. The relief ring also gets its undisplaced
 * geometry recorded so displaceFace can apply the heightmap to it.
 */
async function selectRing(shape, size) {
  const entry = shapeEntry(shape);
  const item = entry && entry.sizes.find((s) => s.size === size);
  if (!item) throw new Error(`no ring ${size} ${shape}`);
  const token = ++RINGS.loadToken;
  const label = `${entry.base}${entry.ridged ? " (ridged)" : ""} · size ${size} (${item.inner_diameter_mm.toFixed(1)} mm)`;
  setRingBadge(`Loading ${label}...`);
  let gltf;
  try {
    // versioned URL: a cached model from an older build must never be paired
    // with this build's design-area data (design rings are rebuilt on their own)
    const version = isReliefRing(shape, size) ? RINGS.reliefVersion : RINGS.catalog.version;
    gltf = await new GLTFLoader().loadAsync(`/rings/${item.file}?v=${version || ""}`);
  } catch (err) {
    setRingBadge(`Failed to load ${label}: ${err.message || err}`, true);
    throw err;
  }
  if (token !== RINGS.loadToken) return; // a newer selection won

  let mesh = null;
  gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  const geo = mesh.geometry;
  geo.computeVertexNormals();

  const relief = isReliefRing(shape, size);
  if (relief) {
    const key = `${shape}/${size}`, info = RINGS.reliefs[key];
    if (info.vertex_count && geo.attributes.position.count !== info.vertex_count) {
      const msg = `${label}: model and design data are from different builds -- rerun tools/prepare_rings.py and reload`;
      setRingBadge(msg, true);
      throw new Error(msg);
    }
    const switched = key !== RINGS.reliefKey;
    RINGS.relief = info;
    RINGS.reliefKey = key;
    const count = geo.attributes.position.count;
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3).fill(1), 3));
    mesh.material = getFaceMaterial();
    state.reliefMesh = mesh;
    const pos = Float32Array.from(geo.attributes.position.array);
    const nrm = Float32Array.from(geo.attributes.normal.array);
    if (info.mode === "tilt") {
      state.reliefBase = prepareReliefZone(pos, nrm, geo.index.array);
      state.reliefBase.area = computeDesignArea(state.reliefBase, state.faceTiltDeg);
      const e = tiltEntry(state.faceTiltDeg);
      setFaceSize(e.width_mm, e.height_mm, e.outline_xz_mm);
    } else {
      state.reliefBase = {
        pos, nrm, n: info.region_vertex_count,
        // the ridge's inner wall: (floor-edge vertex below, height up the wall 0..1) per vertex
        wall: geo.attributes._wall ? geo.attributes._wall.array : null,
        nWall: info.wall_vertex_count || 0,
        // for sliding the ridge's inner wall outward: (signed distance to the floor's edge, outward dx, dz)
        ridge: geo.attributes._ridge ? geo.attributes._ridge.array : null,
      };
      applyRidgeShift(state.reliefBase, info, state.ridgeShiftMm);
    }
    // the heightmap is sized per design area: re-make it for this ring
    if (switched) debouncedRefresh();
  } else {
    mesh.material = getSilverMaterial();
    state.reliefMesh = null;
    state.reliefBase = null;
    state.ringVolumeMm3 = meshVolume(geo.attributes.position.array, geo.index.array);
  }

  // Swap it in, keeping the turntable angle, and free the old one.
  for (const old of [...ringGroup.children]) {
    ringGroup.remove(old);
    old.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
  }
  ringGroup.add(gltf.scene);
  RINGS.current = { shape, size };
  frameRing();
  if (relief) updateMeshHeights();
  updateRingWeight();
  syncRingMenu();
  setRingBadge(label);
}

/** The design area's size: the heightmap's and the crop box's shape follow it. */
function setFaceSize(widthMm, heightMm, outline) {
  state.cfg.face_width_mm = widthMm;
  state.cfg.face_height_mm = heightMm;
  renderCropBox();
  // until a design is picked, show the placeholder, fitted to this area
  if (!state.selectedCandidateId) state.previewCanvas = makeTestHeightmapCanvas(widthMm, heightMm, outline);
}

/**
 * The design area of a "recess" ring (the recessed floor inside a ridge): its
 * vertices are the first region_vertex_count; the design fills it up to the
 * ridge.
 */
function recessArea(base, info, shift = 0) {
  const n = base.n, pos = base.cur || base.pos;
  const w = info.face_width_mm + 2 * shift, h = info.face_height_mm + 2 * shift;
  const x0 = info.x0 - shift, z0 = info.z0 - shift;
  const uv = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    uv[2 * i] = (pos[3 * i] - x0) / w;
    uv[2 * i + 1] = (pos[3 * i + 2] - z0) / h;
  }
  return {
    entry: { width_mm: w, height_mm: h },
    inArea: new Uint8Array(n).fill(1),
    uv,
  };
}

/**
 * How far a point slides outward when the ridge's inner wall is moved out by
 * `shift`, given its signed distance `s` (seen from above) to the floor's edge
 * (negative on the floor). The wall and the fillets at its foot and top move
 * rigidly by `shift`; the floor stretches to follow over the rest of the reach
 * inside, and the ridge's top and outer side compress over the reach outside,
 * fading to nothing so the ring's outside is unchanged. The fades are gentle
 * enough (slope < 1 for shift <= 0.6 mm) that points never pass one another,
 * so no triangle folds.
 */
const RIDGE_RIGID_MM = [-0.45, 0.08]; // floor-edge distances moved rigidly (the fillet, wall and its top fillet)
function ridgeShiftAt(s, shift, reach) {
  const [inner, outer] = reach, [r0, r1] = RIDGE_RIGID_MM;
  if (s <= -inner || s >= outer) return 0;
  if (s < r0) return shift * smoothstep((s + inner) / (r0 + inner));
  if (s <= r1) return shift;
  return shift * (1 - smoothstep((s - r1) / (outer - r1)));
}

/**
 * Slide the ridge's inner wall of a recess ring outward by `shift` mm (see
 * ridgeShiftAt): stores the moved positions as base.cur (displaceFace builds
 * on them), writes them into the mesh, recomputes the floor's UVs and sizes
 * the design area (and heightmap) to the widened floor.
 */
function applyRidgeShift(base, info, shift) {
  const cur = Float32Array.from(base.pos);
  const reach = info.ridge_reach_mm || [0, 0];
  if (base.ridge && shift > 0) {
    const rg = base.ridge, count = cur.length / 3;
    for (let i = 0; i < count; i++) {
      const sh = ridgeShiftAt(rg[3 * i], shift, reach);
      if (!sh) continue;
      cur[3 * i] += rg[3 * i + 1] * sh;
      cur[3 * i + 2] += rg[3 * i + 2] * sh;
    }
  }
  base.cur = cur;
  const attr = state.reliefMesh.geometry.attributes.position;
  attr.array.set(cur);
  attr.needsUpdate = true;
  base.area = recessArea(base, info, shift);
  const sx = (info.face_width_mm + 2 * shift) / info.face_width_mm, sz = (info.face_height_mm + 2 * shift) / info.face_height_mm;
  setFaceSize(info.face_width_mm + 2 * shift, info.face_height_mm + 2 * shift,
    info.outline_xz_mm && info.outline_xz_mm.map(([x, z]) => [x * sx, z * sz]));
  const t = info.ridge_thickness_mm;
  document.getElementById("ridgeShiftVal").textContent =
    `${t ? (t - shift).toFixed(2) + " mm thick" : shift.toFixed(2) + " mm out"} · floor ${(info.face_width_mm + 2 * shift).toFixed(2)} × ${(info.face_height_mm + 2 * shift).toFixed(2)} mm`;
}

/** One-off per load: each top face's tilt from flat, for computeDesignArea. */
function prepareReliefZone(pos, nrm, index) {
  const r = RINGS.relief;
  const nTop = r.top_face_count, n = r.zone_vertex_count;
  const faceTilt = new Float32Array(nTop);
  for (let t = 0; t < nTop; t++) {
    const a = 3 * index[3 * t], b = 3 * index[3 * t + 1], c = 3 * index[3 * t + 2];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    faceTilt[t] = (Math.acos(Math.min(1, Math.max(-1, ny / (Math.hypot(nx, ny, nz) || 1)))) * 180) / Math.PI;
  }
  return { pos, nrm, index, n, nTop, faceTilt };
}

/**
 * The design area for a tilt angle: the top faces tilting less than `deg`
 * and their vertices, with top-down UVs mapping the tabulated bounding box
 * for that angle (the one the backend sizes the heightmap to) onto the
 * heightmap. The rest of the ring isn't moved, so the design ends at the
 * area's edge with a hard edge.
 */
function computeDesignArea(base, deg) {
  const { pos, index, n, nTop, faceTilt } = base;
  const entry = tiltEntry(deg);
  const inArea = new Uint8Array(n);
  for (let t = 0; t < nTop; t++) {
    if (faceTilt[t] >= deg) continue;
    inArea[index[3 * t]] = inArea[index[3 * t + 1]] = inArea[index[3 * t + 2]] = 1;
  }
  const uv = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    uv[2 * i] = (pos[3 * i] - entry.x0) / entry.width_mm;
    uv[2 * i + 1] = (pos[3 * i + 2] - entry.z0) / entry.height_mm;
  }
  return { deg: entry.deg, entry, inArea, uv };
}

/** The tilt design ring's tabulated design area (tilt_table) nearest to an angle. */
function tiltEntry(deg) {
  return RINGS.tiltInfo.tilt_table.reduce((best, e) => (Math.abs(e.deg - deg) < Math.abs(best.deg - deg) ? e : best));
}

/** Volume (mm^3) of a closed triangle mesh: the sum of signed tetrahedra to the origin. */
function meshVolume(pos, index) {
  let v = 0;
  for (let t = 0; t < index.length; t += 3) {
    const a = 3 * index[t], b = 3 * index[t + 1], c = 3 * index[t + 2];
    v += pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
       - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
       + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c]);
  }
  return Math.abs(v) / 6;
}

/** Show the ring's total estimated weight in the report (the ring on show, design at true height). */
function updateRingWeight() {
  const el = document.getElementById("ringWeight");
  if (!el || state.ringVolumeMm3 == null) return;
  const density = state.cfg.silver_density_g_cm3 || 10.37, alloy = state.cfg.silver_alloy || "935";
  const grams = (state.ringVolumeMm3 / 1000) * density;
  el.textContent = `Total ring weight (${alloy} silver): ${grams.toFixed(2)} g (${state.ringVolumeMm3.toFixed(0)} mm³)`;
}

/** Aim the camera at the ring's centre, from the same direction, at a distance that fits it. */
function frameRing() {
  const box = new THREE.Box3().setFromObject(ringGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(dir, size * 2.1);
  controls.minDistance = size * 0.4;
  controls.maxDistance = size * 5;
}

/** Shape / ridged / size menu, built from the catalog. */
function buildRingMenu() {
  const shapeSel = document.getElementById("ringShape");
  const bases = [...new Set(RINGS.catalog.shapes.map((s) => s.base))];
  shapeSel.innerHTML = bases.map((b) => `<option value="${b}">${b}</option>`).join("");
  const pick = () => {
    const base = shapeSel.value;
    const ridged = document.getElementById("ringRidged").checked;
    const entry = RINGS.catalog.shapes.find((s) => s.base === base && s.ridged === ridged)
      || RINGS.catalog.shapes.find((s) => s.base === base);
    const size = document.getElementById("ringSize").value;
    const has = entry.sizes.some((s) => s.size === size);
    selectRing(entry.id, has ? size : entry.sizes[0].size).catch((err) => console.error(err));
  };
  shapeSel.addEventListener("change", pick);
  document.getElementById("ringRidged").addEventListener("change", pick);
  document.getElementById("ringSize").addEventListener("change", pick);
}

/** Reflect the ring on show in the menu (sizes list, current values, relief note). */
function syncRingMenu() {
  const { shape, size } = RINGS.current;
  const entry = shapeEntry(shape);
  document.getElementById("ringShape").value = entry.base;
  document.getElementById("ringRidged").checked = entry.ridged;
  const sizeSel = document.getElementById("ringSize");
  sizeSel.innerHTML = entry.sizes
    .map((s) => `<option value="${s.size}">${s.size} — ${s.inner_diameter_mm.toFixed(1)} mm</option>`)
    .join("");
  sizeSel.value = size;
  const note = document.getElementById("ringReliefNote");
  const onRelief = isReliefRing(shape, size);
  const names = Object.values(RINGS.reliefs).map((r) => {
    const e = shapeEntry(r.shape);
    return `${e.base}${e.ridged ? " (ridged)" : ""} · size ${r.size}`;
  });
  if (names.length && !onRelief) {
    note.hidden = false;
    note.textContent = `Designs are shown on ${names.join(" and ")} only for now.`;
  } else {
    note.hidden = true;
  }
  const info = RINGS.reliefs[`${shape}/${size}`];
  document.getElementById("designTiltGroup").hidden = !(info && info.mode === "tilt");
  document.getElementById("ridgeGroup").hidden = !(info && info.mode === "recess" && info.ridge_reach_mm);
}

/**
 * Set the design area's edge angle: the relief ring's top that tilts less than
 * `deg`. The heightmap and crop box take that area's size (the tabulated box);
 * the 3D area is recomputed at most once per frame while dragging. `reprocess`
 * asks the backend for a heightmap of the new size (on slider release).
 */
let tiltFrame = 0;
function setDesignTilt(deg, reprocess) {
  state.faceTiltDeg = deg;
  const e = tiltEntry(deg);
  document.getElementById("designTiltVal").textContent = `${e.deg.toFixed(1)}° (${e.width_mm.toFixed(2)} × ${e.height_mm.toFixed(2)} mm)`;
  const onTiltRing = RINGS.relief && RINGS.relief.mode === "tilt";
  if (!onTiltRing) return; // the angle applies to the tilt ring's design area only
  setFaceSize(e.width_mm, e.height_mm, e.outline_xz_mm);
  if (state.reliefBase && !tiltFrame) {
    tiltFrame = requestAnimationFrame(() => {
      tiltFrame = 0;
      state.reliefBase.area = computeDesignArea(state.reliefBase, state.faceTiltDeg);
      updateMeshHeights();
    });
  }
  if (reprocess) debouncedRefresh();
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
 * Apply the heightmap to the relief ring (the design area comes from
 * computeDesignArea / recessArea):
 *
 * 1. Design-area vertices are pushed out along their normal (straight up) by
 *    the heightmap at their top-down UV. Nothing outside the area moves, so
 *    the design ends at the area's edge with a hard edge.
 * 2. On a "recess" ring (the design fills the floor inside a ridge) the floor
 *    moves straight up/down instead, lifted so the design's highest point is
 *    level with the top of the ridge's inner wall, and the wall stretches so
 *    its foot follows the floor edge below it and its top stays at the ridge.
 *
 * UV u runs along +X and v along +Z with image row 0 at -Z. The area's
 * recesses get the oxidized patina (see PATINA).
 */
function displaceFace() {
  const { nrm, n, area, wall, nWall } = state.reliefBase;
  const basePos = state.reliefBase.cur || state.reliefBase.pos; // recess rings: with the ridge wall slid out
  // recess rings move the floor straight up/down, so its edge stays directly
  // under the ridge's wall, which then follows it (below)
  const vertical = RINGS.relief.mode === "recess";
  // ...and the design's highest point always sits level with the top of the
  // ridge's inner wall, at any exaggeration (which only deepens what's below it)
  const recessLift = vertical ? RINGS.relief.wall_top_y_mm - RINGS.relief.floor_y_mm : 0;
  const { inArea, uv, entry } = area;
  const src = state.previewCanvas;
  const w = src.width, h = src.height;
  const { data } = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h);
  const reliefMax = state.cfg.relief_max_mm, exag = state.exaggeration;

  const raw = new Float32Array(w * h);
  let top = 0; // the design's highest point
  for (let i = 0; i < w * h; i++) { raw[i] = data[i * 4] / 255; if (raw[i] > top) top = raw[i]; } // grayscale, R channel
  const blurred = boxBlur(raw, w, h, Math.max(1, Math.round((PATINA.radiusMm / entry.width_mm) * w)));
  // The preview exaggeration grows the design downwards: its highest point
  // stays at its true height and everything below it sinks `exag` times
  // deeper (at 1x this is just the true height). With "exaggerate upward"
  // the base stays put instead and the peaks rise `exag` times higher.
  const anchor = state.exaggerateUp ? 0 : top;
  const displaceMm = (height) => reliefMax * (anchor + (height - anchor) * exag);

  const geo = state.reliefMesh.geometry;
  const pos = geo.attributes.position;
  const col = geo.attributes.color;
  // the same shape at the design's true height (exaggeration 1), for the ring's weight
  const truePos = Float32Array.from(basePos);
  const displace1 = (height) => reliefMax * height;
  for (let i = 0; i < n; i++) {
    let d = 0, d1 = 0, shade = 1;
    if (inArea[i]) {
      const u = uv[2 * i], v = uv[2 * i + 1];
      const height = sampleBilinear(raw, w, h, u, v);
      const cavity = Math.max(0, sampleBilinear(blurred, w, h, u, v) - height);
      shade = 1 - PATINA.darkness * smoothstep(cavity * PATINA.strength);
      d = vertical ? recessLift + reliefMax * (anchor + (height - anchor) * exag - top) : displaceMm(height);
      d1 = vertical ? recessLift + reliefMax * (height - top) : displace1(height);
    }
    if (vertical) {
      pos.setXYZ(i, basePos[3 * i], basePos[3 * i + 1] + d, basePos[3 * i + 2]);
      truePos[3 * i + 1] += d1;
    } else {
      pos.setXYZ(i, basePos[3 * i] + nrm[3 * i] * d, basePos[3 * i + 1] + nrm[3 * i + 1] * d, basePos[3 * i + 2] + nrm[3 * i + 2] * d);
      truePos[3 * i] += nrm[3 * i] * d1; truePos[3 * i + 1] += nrm[3 * i + 1] * d1; truePos[3 * i + 2] += nrm[3 * i + 2] * d1;
    }
    col.setXYZ(i, shade, shade, shade);
  }
  // 2. recess rings: stretch the ridge's inner wall so its foot follows the
  // floor edge below it (down where the design sinks it, up where it rises)
  // while its top at the ridge stays put
  if (vertical && wall) {
    for (let i = n; i < n + nWall; i++) {
      const e = wall[2 * i];
      if (e < 0) continue;
      const floorDy = pos.getY(e) - basePos[3 * e + 1];
      pos.setY(i, basePos[3 * i + 1] + floorDy * (1 - wall[2 * i + 1]));
      truePos[3 * i + 1] = basePos[3 * i + 1] + (truePos[3 * e + 1] - basePos[3 * e + 1]) * (1 - wall[2 * i + 1]);
    }
  }
  state.ringVolumeMm3 = meshVolume(truePos, geo.index.array);
  updateRingWeight();
  pos.needsUpdate = true;
  col.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
}

function updateMeshHeights() {
  if (!state.previewCanvas || !state.reliefBase) return;
  displaceFace();
}

// ---------------------------------------------------------------------------
// Hard-coded test heightmap (Milestone 1 — shown until a real design loads)
// ---------------------------------------------------------------------------

/**
 * The placeholder design shown until one is picked, fitted to the design
 * area: concentric copies of the area's outline (mm, centred on its box),
 * brighter towards the middle, so the outermost fills the area right up to
 * its edge (on a ridged ring, to the ridge's walls). Circles if no outline.
 */
function makeTestHeightmapCanvas(widthMm, heightMm, outline) {
  const w = 460, h = Math.max(1, Math.round((460 * heightMm) / widthMm));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  const steps = [1.0, 0.78, 0.56, 0.34];
  steps.forEach((f, i) => {
    const v = Math.round(255 * ((i + 1) / steps.length));
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.beginPath();
    if (outline && outline.length > 2) {
      outline.forEach(([x, z], k) => {
        const px = (0.5 + (f * x) / widthMm) * w, py = (0.5 + (f * z) / heightMm) * h;
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
    } else {
      ctx.arc(w / 2, h / 2, (Math.min(w, h) / 2) * f, 0, Math.PI * 2);
    }
    ctx.fill();
  });
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
    relief_ring: RINGS.reliefKey,
    face_tilt_deg: state.faceTiltDeg,
    ridge_shift_mm: RINGS.relief && RINGS.relief.mode === "recess" ? state.ridgeShiftMm : null,
    remove_background: document.getElementById("removeBg").checked,
    height_source: document.getElementById("heightSource").value,
    depth_detail: parseFloat(document.getElementById("depthDetail").value),
    bas_relief: parseFloat(document.getElementById("basRelief").value),
    denoise: document.getElementById("denoise").checked,
    upscale: document.getElementById("upscale").checked,
    smooth_mm: parseFloat(document.getElementById("smoothMm").value),
    outline_strength: parseFloat(document.getElementById("outlineStrength").value),
    hatch_strength: parseFloat(document.getElementById("hatchStrength").value),
    hatch_spacing_mm: parseFloat(document.getElementById("hatchSpacing").value),
    hatch_angle_deg: parseFloat(document.getElementById("hatchAngle").value),
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
    <div id="ringWeight"></div>
    <div>Relief alone: ${report.estimated_weight_g.toFixed(3)} g</div>
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
  updateRingWeight();
}

async function refreshFromBackend() {
  if (!state.selectedCandidateId) return;
  const slow = [];
  if (document.getElementById("removeBg").checked) slow.push("finding the subject");
  if (document.getElementById("heightSource").value === "depth") slow.push("estimating depth");
  if (document.getElementById("upscale").checked) slow.push("upscaling");
  setStatus("processStatus", slow.length
    ? `Processing (${slow.join(", ")}: the first time for an image can take a while)...`
    : "Processing...");
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

// ---------------------------------------------------------------------------
// Design with AI: re-sculpt the selected image as a relief (POST /api/relief)
// ---------------------------------------------------------------------------

const AI = { options: null, style: null, additions: [], busy: false };

async function initAI() {
  try {
    AI.options = await (await fetch("/api/relief/options")).json();
  } catch (err) {
    console.warn("AI relief options unavailable", err);
    return;
  }
  AI.style = AI.options.default_style;
  const fid = document.getElementById("aiFidelity");
  fid.value = AI.options.fidelity_default;
  const showFid = () => {
    const f = parseFloat(fid.value);
    document.getElementById("aiFidelityVal").textContent = f >= 0.75 ? "(close)" : f >= 0.55 ? "(balanced)" : "(creative)";
  };
  fid.addEventListener("input", showFid);
  showFid();

  const styles = document.getElementById("aiStyles");
  for (const st of AI.options.styles) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = st.label;
    b.dataset.id = st.id;
    b.addEventListener("click", () => { AI.style = st.id; renderAIChoices(); });
    styles.appendChild(b);
  }
  const strip = document.getElementById("aiAdditions");
  for (const a of AI.options.additions) {
    const card = document.createElement("div");
    card.className = "addition-card";
    card.dataset.id = a.id;
    card.title = a.label;
    const img = document.createElement("img");
    img.src = a.thumb;
    img.alt = a.label;
    const label = document.createElement("span");
    label.textContent = a.label;
    card.append(img, label);
    card.addEventListener("click", () => toggleAddition(a.id));
    strip.appendChild(card);
  }
  document.getElementById("aiText").addEventListener("input", renderAIChoices);
  document.getElementById("aiText").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); generateRelief(); }
  });
  document.getElementById("aiGenerateBtn").addEventListener("click", generateRelief);
  renderAIChoices();
  refreshUsage();
}

/** Pick or drop an addition; only one per group (e.g. one background) at a time. */
function toggleAddition(id) {
  if (AI.additions.includes(id)) {
    AI.additions = AI.additions.filter((x) => x !== id);
  } else {
    const group = AI.options.additions.find((a) => a.id === id).group;
    AI.additions = AI.additions.filter((x) => AI.options.additions.find((a) => a.id === x).group !== group);
    AI.additions.push(id);
  }
  renderAIChoices();
}

/** Highlight the picked style/additions, show them as tags in the chatbox, and price the request. */
function renderAIChoices() {
  if (!AI.options) return;
  document.querySelectorAll("#aiStyles .chip").forEach((b) => b.classList.toggle("on", b.dataset.id === AI.style));
  document.querySelectorAll("#aiAdditions .addition-card").forEach((c) => c.classList.toggle("on", AI.additions.includes(c.dataset.id)));
  const chips = document.getElementById("aiChips");
  chips.innerHTML = "";
  const style = AI.options.styles.find((s) => s.id === AI.style);
  const st = document.createElement("span");
  st.className = "tag style";
  st.textContent = style.label;
  chips.appendChild(st);
  for (const id of AI.additions) {
    const a = AI.options.additions.find((x) => x.id === id);
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.append(a.label);
    const x = document.createElement("button");
    x.textContent = "×";
    x.title = "Remove";
    x.addEventListener("click", () => toggleAddition(id));
    tag.appendChild(x);
    chips.appendChild(tag);
  }
  const typed = document.getElementById("aiText").value.trim().length > 0;
  const n = typed ? AI.options.images_with_text : AI.options.images_default;
  const usd = n * (typed ? AI.options.price_usd_variation : AI.options.price_usd_single);
  const price = AI.options.provider === "mock" ? "free offline preview" : `~$${usd.toFixed(usd < 0.1 ? 3 : 2)}`;
  const btn = document.getElementById("aiGenerateBtn");
  btn.textContent = AI.busy ? "Generating…" : `Generate ${n === 1 ? "1 image" : n + " variations"} · ${price}`;
  btn.disabled = AI.busy || !state.selectedCandidateId;
}

function addChatMessage(text, kind) {
  const log = document.getElementById("aiChatLog");
  const div = document.createElement("div");
  div.className = "chat-msg " + kind;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function generateRelief() {
  if (AI.busy || !state.selectedCandidateId || !AI.options) return;
  const textEl = document.getElementById("aiText");
  const text = textEl.value.trim();
  const style = AI.options.styles.find((s) => s.id === AI.style).label;
  const extras = AI.additions.map((id) => AI.options.additions.find((a) => a.id === id).label);
  addChatMessage(`You: ${[style, ...extras].join(" + ")}${text ? " — “" + text + "”" : ""}`, "you");
  AI.busy = true;
  renderAIChoices();
  setStatus("aiStatus", "Sculpting your relief… this takes a few seconds per image.");
  try {
    const res = await fetch("/api/relief", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: state.selectedCandidateId, style: AI.style, additions: AI.additions, text,
                             fidelity: parseFloat(document.getElementById("aiFidelity").value) }),
    });
    if (!res.ok) {
      let msg = await res.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const out = await res.json();
    addChatMessage(`${out.candidates.length === 1 ? "1 relief" : out.candidates.length + " reliefs"} added below — the first is on the ring.`, "ai");
    textEl.value = "";
    renderUsage(out.usage);
    state.candidates = [...state.candidates, ...out.candidates];
    renderGallery(state.candidates);
    // a relief picture is lit sculpture, so its shape is best read with AI depth
    const hs = document.getElementById("heightSource");
    if (hs.value !== "depth") { hs.value = "depth"; hs.dispatchEvent(new Event("change")); }
    const first = out.candidates[0];
    await selectCandidate(first.candidate_id, document.querySelector(`#gallery img[data-candidate-id="${first.candidate_id}"]`));
    setStatus("aiStatus", "");
  } catch (err) {
    console.error(err);
    addChatMessage(`Couldn't generate: ${err.message}`, "err");
    setStatus("aiStatus", "Generation failed.", true);
  } finally {
    AI.busy = false;
    renderAIChoices();
  }
}

async function refreshUsage() {
  try {
    renderUsage(await (await fetch("/api/usage")).json());
  } catch (err) {
    console.warn("usage unavailable", err);
  }
}

function renderUsage(u) {
  const el = document.getElementById("aiUsage");
  if (!u) return;
  if (u.provider === "mock") {
    el.innerHTML = `AI relief: <b>offline preview</b> (no API key) · ${u.images} free images made`;
    return;
  }
  const src = u.provider === "fal" ? "fal.ai" : "Stability AI";
  const bal = u.balance_credits == null ? "" : ` · balance <b>${Math.round(u.balance_credits)}</b> credits`;
  el.innerHTML = `${src} usage today: <b>${u.today.images}</b> images · <b>$${u.today.usd.toFixed(2)}</b>`
    + ` · all time $${u.usd.toFixed(2)} (${u.images} images)${bal}`;
  if (AI.options && AI.options.models) {
    el.title = `Single images: ${AI.options.models.single}\n4 variations: ${AI.options.models.variations}`;
  }
}

async function selectCandidate(candidateId, imgEl) {
  state.selectedCandidateId = candidateId;
  document.querySelectorAll("#gallery img").forEach((el) => el.classList.remove("selected"));
  if (imgEl) imgEl.classList.add("selected");
  document.getElementById("cropSection").hidden = false;
  document.getElementById("paramsSection").hidden = false;
  document.getElementById("aiSection").hidden = false;
  renderAIChoices();
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
  document.getElementById("removeBg").addEventListener("change", debouncedRefresh);

  // Image enhancement (app/processing.py, app/enhance.py): ring-agnostic
  ["depthDetail", "basRelief", "smoothMm", "outlineStrength", "hatchStrength", "hatchSpacing", "hatchAngle"].forEach((id) => {
    wireSliderDisplay(id);
    document.getElementById(id).addEventListener("change", debouncedRefresh);
  });
  ["heightSource", "denoise", "upscale"].forEach((id) => document.getElementById(id).addEventListener("change", debouncedRefresh));
  const syncEnhanceGroups = () => {
    document.getElementById("depthDetailGroup").hidden = document.getElementById("heightSource").value !== "depth";
    document.getElementById("hatchGroup").hidden = parseFloat(document.getElementById("hatchStrength").value) <= 0;
  };
  document.getElementById("heightSource").addEventListener("change", syncEnhanceGroups);
  document.getElementById("hatchStrength").addEventListener("input", syncEnhanceGroups);
  syncEnhanceGroups();
  document.getElementById("flipH").addEventListener("change", debouncedRefresh);
  document.getElementById("flipV").addEventListener("change", debouncedRefresh);

  const tilt = document.getElementById("designTilt");
  tilt.addEventListener("input", () => setDesignTilt(parseFloat(tilt.value), false));
  tilt.addEventListener("change", () => setDesignTilt(parseFloat(tilt.value), true));
  const ridge = document.getElementById("ridgeShift");
  ridge.value = state.ridgeShiftMm;
  const onRidge = (reprocess) => {
    state.ridgeShiftMm = parseFloat(ridge.value);
    const base = state.reliefBase, info = RINGS.relief;
    if (!base || !info || info.mode !== "recess") return;
    applyRidgeShift(base, info, state.ridgeShiftMm);
    if (!tiltFrame) tiltFrame = requestAnimationFrame(() => { tiltFrame = 0; updateMeshHeights(); });
    if (reprocess) debouncedRefresh();
  };
  ridge.addEventListener("input", () => onRidge(false));
  ridge.addEventListener("change", () => onRidge(true));

  wireSliderDisplay("exaggeration", "x");
  document.getElementById("exaggeration").addEventListener("input", (e) => {
    state.exaggeration = parseFloat(e.target.value);
    updateMeshHeights();
  });
  document.getElementById("exaggerateUp").addEventListener("change", (e) => {
    state.exaggerateUp = e.target.checked;
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

  // (loading the default ring already made a placeholder fitted to its design area)
  if (!state.previewCanvas) state.previewCanvas = makeTestHeightmapCanvas(cfg.face_width_mm, cfg.face_height_mm, cfg.face_outline_mm);
  updateMeshHeights();

  wireUI(cfg);
  initAI();
}

main();
