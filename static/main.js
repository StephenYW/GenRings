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
 * Square for now, see relief.json) carries the relief: its vertices
 * 0..region_vertex_count-1 are its top face (the flat top and its rounded
 * edge, up to where it rolls over into the shoulders), with top-down UVs and a
 * per-vertex weight that eases the relief out at the face's edge.
 */
const RINGS = {
  catalog: null,
  relief: null,        // relief.json
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
    fetch("/rings/catalog.json").then((r) => r.json()),
    fetch("/rings/relief.json").then((r) => r.json()).catch(() => null),
  ]);
  RINGS.catalog = catalog;
  RINGS.relief = relief;
  buildRingMenu();
  const d = catalog.default;
  await selectRing(d.shape, d.size);
}

function shapeEntry(id) {
  return RINGS.catalog.shapes.find((s) => s.id === id);
}

function isReliefRing(shape, size) {
  return RINGS.relief && RINGS.relief.shape === shape && RINGS.relief.size === size;
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
    gltf = await new GLTFLoader().loadAsync(`/rings/${item.file}`);
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
    const n = RINGS.relief.region_vertex_count;
    const count = geo.attributes.position.count;
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3).fill(1), 3));
    mesh.material = getFaceMaterial();
    state.reliefMesh = mesh;
    state.reliefBase = {
      pos: Float32Array.from(geo.attributes.position.array),
      nrm: Float32Array.from(geo.attributes.normal.array),
      uv: geo.attributes.uv.array,
      weight: geo.attributes._weight ? geo.attributes._weight.array : new Float32Array(n).fill(1),
      n,
    };
  } else {
    mesh.material = getSilverMaterial();
    state.reliefMesh = null;
    state.reliefBase = null;
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
  syncRingMenu();
  setRingBadge(label);
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
  const r = RINGS.relief;
  if (r && !isReliefRing(shape, size)) {
    note.hidden = false;
    note.textContent = `Designs are shown on ${r.shape} · size ${r.size} only for now.`;
  } else {
    note.hidden = true;
  }
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
 * Apply the heightmap to the relief ring. Each vertex of its top face is
 * pushed out along its (undisplaced) surface normal by the heightmap sampled
 * at its top-down UV -- straight up on the flat top, angled on the rounded
 * edge -- times its edge weight, which eases the relief out right at the
 * face's edge so the plain shoulders below meet it without a step.
 * UV u runs along +X and v along +Z with image row 0 at -Z. Recesses get the
 * oxidized patina (see PATINA).
 */
function displaceFace() {
  const { pos: basePos, nrm, uv, weight, n } = state.reliefBase;
  const src = state.previewCanvas;
  const w = src.width, h = src.height;
  const { data } = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h);
  const scaleMm = state.cfg.relief_max_mm * state.exaggeration;

  const raw = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) raw[i] = data[i * 4] / 255; // grayscale, R channel
  const blurred = boxBlur(raw, w, h, Math.max(1, Math.round((PATINA.radiusMm / state.cfg.face_width_mm) * w)));

  const geo = state.reliefMesh.geometry;
  const pos = geo.attributes.position;
  const col = geo.attributes.color;
  for (let i = 0; i < n; i++) {
    const u = uv[2 * i], v = uv[2 * i + 1];
    const height = sampleBilinear(raw, w, h, u, v);
    const d = height * scaleMm * weight[i];
    pos.setXYZ(i, basePos[3 * i] + nrm[3 * i] * d, basePos[3 * i + 1] + nrm[3 * i + 1] * d, basePos[3 * i + 2] + nrm[3 * i + 2] * d);
    const cavity = Math.max(0, sampleBilinear(blurred, w, h, u, v) - height);
    const c = 1 - PATINA.darkness * smoothstep(cavity * PATINA.strength) * weight[i];
    col.setXYZ(i, c, c, c);
  }
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

function makeTestHeightmapCanvas(cfg) {
  const w = 460, h = Math.round((460 * cfg.face_height_mm) / cfg.face_width_mm);
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
