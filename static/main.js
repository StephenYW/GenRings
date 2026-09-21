import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  cfg: null,
  selectedCandidateId: null,
  candidates: [],
  previewCanvas: null,     // offscreen canvas holding the current 8-bit preview (grayscale)
  exaggeration: 3,
  reliefMesh: null,
  gridNX: 400,
  gridNY: 340,
  debounceTimer: null,
};

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------

let renderer, scene, camera, controls;

function initThree(cfg) {
  const canvas = document.getElementById("three-canvas");
  const viewer = document.getElementById("viewer");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1c20);

  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

  const aspect = viewer.clientWidth / viewer.clientHeight;
  camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 500);
  const maxDim = Math.max(cfg.face_width_mm, cfg.face_height_mm);
  camera.position.set(0, maxDim * 1.1, maxDim * 1.35);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.minDistance = maxDim * 0.6;
  controls.maxDistance = maxDim * 6;
  controls.maxPolarAngle = Math.PI * 0.49;

  // Key light to add a specular highlight beyond the environment map.
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(maxDim, maxDim * 2, maxDim);
  scene.add(keyLight);
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));

  buildBezel(cfg);
  buildReliefMesh(cfg);

  onResize();
  window.addEventListener("resize", onResize);

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function onResize() {
  const viewer = document.getElementById("viewer");
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

const METAL_COLOR = 0xd7d7da; // polished silver

function buildBezel(cfg) {
  const rim = 1.6;
  const slabThickness = 1.6;
  const geo = new THREE.BoxGeometry(
    cfg.face_width_mm + rim * 2,
    slabThickness,
    cfg.face_height_mm + rim * 2
  );
  const mat = new THREE.MeshStandardMaterial({
    color: METAL_COLOR,
    metalness: 1.0,
    roughness: 0.22,
  });
  const slab = new THREE.Mesh(geo, mat);
  slab.position.y = -slabThickness / 2 - 0.001;
  scene.add(slab);
}

function buildReliefMesh(cfg) {
  const nx = state.gridNX;
  const ny = state.gridNY;
  const geo = new THREE.PlaneGeometry(cfg.face_width_mm, cfg.face_height_mm, nx - 1, ny - 1);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardMaterial({
    color: METAL_COLOR,
    metalness: 1.0,
    roughness: 0.08,
  });

  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);
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

function updateMeshHeights() {
  if (!state.previewCanvas || !state.reliefMesh) return;
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
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, preset, n_candidates: 4 }),
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
// UI wiring
// ---------------------------------------------------------------------------

function currentParams() {
  return {
    invert: document.getElementById("invert").checked,
    gamma: 1.0,
    contrast: 1.0,
    blur_mm: parseFloat(document.getElementById("blur").value),
    levels: parseInt(document.getElementById("levels").value, 10),
    min_feature_mm: parseFloat(document.getElementById("minFeature").value),
    relief_height_mm: parseFloat(document.getElementById("reliefHeight").value),
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

function selectCandidate(candidateId, imgEl) {
  state.selectedCandidateId = candidateId;
  document.querySelectorAll("#gallery img").forEach((el) => el.classList.remove("selected"));
  if (imgEl) imgEl.classList.add("selected");
  document.getElementById("paramsSection").hidden = false;
  applyPresetDefaults(state.cfg);
  refreshFromBackend();
}

function wireUI(cfg) {
  populatePresets(cfg);

  ["reliefHeight", "levels", "blur", "minFeature"].forEach((id) => {
    wireSliderDisplay(id);
    document.getElementById(id).addEventListener("change", debouncedRefresh);
  });
  document.getElementById("invert").addEventListener("change", debouncedRefresh);

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
    setStatus("generateStatus", "Generating candidates...");
    try {
      const result = await generateCandidates(prompt, preset);
      state.candidates = result.candidates;
      renderGallery(result.candidates);
      setStatus("generateStatus", `${result.candidates.length} candidates ready. Pick one below.`);
      if (result.warnings && result.warnings.length) {
        setStatus("generateStatus", result.warnings.join(" "));
      }
      if (result.candidates.length) {
        const firstImg = document.querySelector("#gallery img");
        selectCandidate(result.candidates[0].candidate_id, firstImg);
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

  initThree(cfg);

  state.previewCanvas = makeTestHeightmapCanvas(cfg);
  updateMeshHeights();

  wireUI(cfg);
}

main();
