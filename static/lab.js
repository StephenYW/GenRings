// Prompt lab (see app/prompt_lab.py): test images, prompt versions, runs and reviews.

const $ = (id) => document.getElementById(id);
const S = {
  opts: null,
  sources: [],
  editingSource: null,
  versions: [],
  version: null,          // the prompt version open in the editor (full JSON)
  runs: [],
  run: null,              // the run open for review (with items)
  compare: null,          // another run to compare with
  itemIdx: -1,            // index into visibleItems()
  pollTimer: 0,
};

// --- helpers ---------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(e.dataset, v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c);
  return e;
}

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { "Content-Type": "application/json" };
    delete init.json;
  }
  const res = await fetch(path, init);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    let msg = typeof data === "string" ? data : data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
    throw new Error(msg || `${res.status} ${res.statusText}`);
  }
  return data;
}

const fileUrl = (path, w) => `/api/lab/files/${path.split("/").map(encodeURIComponent).join("/")}${w ? `?w=${w}` : ""}`;
const catLabel = (id) => (S.opts.categories.find((c) => c.id === id) || { label: id }).label;
const money = (x) => (x == null ? "—" : `$${x.toFixed(x < 1 ? 3 : 2)}`);
const tagLabel = (t) => t.replace(/_/g, " ");

function setStatus(id, msg, error = false) {
  const s = $(id);
  s.textContent = msg || "";
  s.classList.toggle("error", !!error);
}

function openLightbox(src) {
  const lb = $("lightbox");
  lb.querySelector("img").src = src;
  lb.hidden = false;
}
$("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });

// --- tabs ------------------------------------------------------------------------

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name !== "images") $("srcEditor").hidden = true;
  if (name === "prompts") loadVersions();
  if (name === "runs") { loadVersions(); loadRuns(); }
  try { localStorage.setItem("lab.tab", name); } catch { /* storage unavailable */ }
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// --- test images -------------------------------------------------------------------

async function loadSources() {
  S.sources = (await api("/api/lab/sources")).sources;
  renderSources();
  renderRunEstimate();
}

function renderSources() {
  const prog = $("catProgress");
  prog.replaceChildren();
  for (const c of S.opts.categories) {
    const rows = S.sources.filter((s) => s.category === c.id);
    const held = rows.filter((s) => s.split === "holdout").length;
    prog.append(el("span", {
      class: `chip stat${rows.length >= S.opts.target_per_category ? " full" : ""}`,
      title: `${rows.length - held} tune, ${held} holdout`,
      text: `${c.label} ${rows.length}/${S.opts.target_per_category}`,
    }));
  }
  const cat = $("srcFilterCat").value, split = $("srcFilterSplit").value;
  const grid = $("srcGrid");
  grid.replaceChildren();
  const rows = S.sources.filter((s) => (!cat || s.category === cat) && (!split || s.split === split));
  if (!rows.length) grid.append(el("p", { class: "muted", text: "No test images yet." }));
  for (const s of rows) {
    grid.append(el("div", { class: "tile", title: [s.origin, s.licence, s.tags].filter(Boolean).join(" · "), onclick: () => editSource(s.id) },
      el("img", { src: fileUrl(`sources/${s.file}`, 256), loading: "lazy", alt: s.id }),
      el("div", { class: "cap" }, el("span", { text: s.id }), el("span", { class: `split-${s.split}`, text: s.split }))));
  }
}

async function uploadFiles(files) {
  const images = [...files].filter((f) => /^image\/(png|jpe?g|webp)$/.test(f.type));
  if (!images.length) { setStatus("upStatus", "No PNG, JPEG or WebP images in that selection.", true); return; }
  const fd = new FormData();
  for (const f of images) fd.append("files", f);
  fd.append("category", $("upCategory").value);
  fd.append("split", $("upSplit").value);
  fd.append("origin", $("upOrigin").value);
  fd.append("licence", $("upLicence").value);
  fd.append("tags", $("upTags").value);
  setStatus("upStatus", `Uploading ${images.length} image${images.length > 1 ? "s" : ""}…`);
  try {
    const out = await api("/api/lab/sources", { method: "POST", body: fd });
    const msg = `Added ${out.added.length} to ${catLabel($("upCategory").value)}.`;
    setStatus("upStatus", out.errors.length ? `${msg} Skipped: ${out.errors.join("; ")}` : msg, out.errors.length > 0);
    await loadSources();
  } catch (err) {
    setStatus("upStatus", err.message, true);
  }
}

function editSource(id) {
  const s = S.sources.find((r) => r.id === id);
  if (!s) return;
  S.editingSource = id;
  $("srcEdTitle").textContent = `${s.id} · ${catLabel(s.category)}`;
  $("srcEdImg").src = fileUrl(`sources/${s.file}`, 720);
  $("srcEdSplit").value = s.split;
  $("srcEdOrigin").value = s.origin;
  $("srcEdLicence").value = s.licence;
  $("srcEdTags").value = s.tags;
  $("srcEdNotes").value = s.notes;
  setStatus("srcEdStatus", "");
  $("srcEditor").hidden = false;
}

async function saveSource() {
  try {
    await api(`/api/lab/sources/${S.editingSource}`, { method: "PATCH", json: {
      split: $("srcEdSplit").value, origin: $("srcEdOrigin").value, licence: $("srcEdLicence").value,
      tags: $("srcEdTags").value, notes: $("srcEdNotes").value,
    } });
    setStatus("srcEdStatus", "Saved.");
    await loadSources();
  } catch (err) { setStatus("srcEdStatus", err.message, true); }
}

async function deleteSource() {
  if (!confirm(`Delete ${S.editingSource}? Results already made from it stay in their runs.`)) return;
  try {
    await api(`/api/lab/sources/${S.editingSource}`, { method: "DELETE" });
    $("srcEditor").hidden = true;
    await loadSources();
  } catch (err) { setStatus("srcEdStatus", err.message, true); }
}

function wireImages() {
  for (const c of S.opts.categories) {
    $("upCategory").append(el("option", { value: c.id, text: c.label }));
    $("srcFilterCat").append(el("option", { value: c.id, text: c.label }));
  }
  $("upFiles").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
  const dz = $("dropZone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); uploadFiles(e.dataTransfer.files); });
  $("srcFilterCat").addEventListener("change", renderSources);
  $("srcFilterSplit").addEventListener("change", renderSources);
  $("srcEdClose").addEventListener("click", () => { $("srcEditor").hidden = true; });
  $("srcEdSave").addEventListener("click", saveSource);
  $("srcEdDelete").addEventListener("click", deleteSource);
  $("srcEdImg").addEventListener("click", () => {
    const s = S.sources.find((r) => r.id === S.editingSource);
    if (s) openLightbox(fileUrl(`sources/${s.file}`));
  });
}

// --- prompt versions -----------------------------------------------------------------

const TEMPLATE_FIELDS = [
  ["convert", "Main instruction", 2],
  ["keep_all", "Keep everything (no background option)", 3],
  ["keep_subject", "Keep the subject (with a background option)", 3],
  ["except_changes", "“Except changes” phrase", 1],
  ["casting", "Simplify for casting", 3],
  ["rules", "Rules (end of prompt)", 3],
  ["negative", "Negative prompt", 2],
  ["describe", "Description (only non-instruction models, e.g. Stability)", 2],
];

async function loadVersions(select) {
  S.versions = (await api("/api/lab/prompts")).versions;
  const list = $("versionList");
  list.replaceChildren();
  for (const v of [...S.versions].reverse()) {
    list.append(el("div", { class: `ver${S.version && S.version.version === v.version ? " on" : ""}`, onclick: () => openVersion(v.version) },
      el("div", { class: "row-between" }, el("b", { text: v.version }),
        v.locked ? el("span", { class: "muted", text: `${v.runs.length} run${v.runs.length > 1 ? "s" : ""}` }) : el("span", { class: "muted", text: "draft" })),
      v.based_on ? el("div", { class: "muted", text: `from ${v.based_on}` }) : null,
      el("div", { class: "notes", text: v.notes || "" })));
  }
  const runSel = $("runVersion"), keep = runSel.value;
  runSel.replaceChildren(...[...S.versions].reverse().map((v) => el("option", { value: v.version, text: `${v.version}${v.notes ? ` · ${v.notes.slice(0, 50)}` : ""}` })));
  if (keep && S.versions.some((v) => v.version === keep)) runSel.value = keep;
  const want = select || (S.version && S.version.version) || S.versions[S.versions.length - 1].version;
  if (!S.version || select || S.version.version !== want) await openVersion(want);
}

function templateBox(id, label, rows, value, locked) {
  return el("div", { class: "tmpl" }, el("label", { text: label },
    el("textarea", { id, rows, disabled: locked }, value || "")));
}

async function openVersion(name) {
  const v = await api(`/api/lab/prompts/${name}`);
  S.version = v;
  const locked = v.locked;
  $("verTitle").textContent = `${v.version}${v.based_on ? ` (from ${v.based_on})` : ""}`;
  $("verLock").hidden = !locked;
  $("verSave").disabled = locked;
  $("verNotes").value = v.notes || "";
  $("verNotes").disabled = locked;
  setStatus("verStatus", locked ? `Used by ${v.runs.join(", ")}. Make a new version to change it.` : "");

  const model = $("verModel");
  model.replaceChildren();
  const models = S.opts.models.length ? S.opts.models : [{ id: "", label: `(${S.opts.provider} provider)`, price_usd: 0 }];
  for (const m of models) model.append(el("option", { value: m.id, text: m.id ? `${m.label} · ~${money(m.price_usd)}` : m.label }));
  if (v.settings.model && !models.some((m) => m.id === v.settings.model)) model.append(el("option", { value: v.settings.model, text: v.settings.model }));
  model.value = v.settings.model || models[0].id;
  model.disabled = locked;
  const style = $("verStyle");
  style.replaceChildren(...S.opts.styles.map((s) => el("option", { value: s.id, text: s.label })));
  style.value = v.settings.style;
  style.disabled = locked;
  $("verFidelity").value = v.settings.fidelity;
  $("verFidelity").disabled = locked;
  const adds = $("verAdditions");
  adds.replaceChildren();
  const picked = new Set(v.settings.additions || []);
  for (const a of S.opts.additions) {
    adds.append(el("button", { class: `chip${picked.has(a.id) ? " on" : ""}`, dataset: { id: a.id }, title: a.group, disabled: locked,
      onclick: (e) => e.currentTarget.classList.toggle("on") }, a.label));
  }

  const t = v.templates;
  $("verTemplates").replaceChildren(...TEMPLATE_FIELDS.map(([k, label, rows]) => templateBox(`tpl-${k}`, label, rows, t[k], locked)));
  $("verImageTypes").replaceChildren(...Object.entries(t.image_types).map(([k, val]) =>
    templateBox(`it-${k}`, (S.opts.categories.find((c) => c.id === k) || { label: k }).label + (k === "auto" ? " (Auto: not a lab category)" : ""), 2, val, locked)));
  $("verStyles").replaceChildren(...Object.entries(t.styles).map(([k, val]) => el("div", { class: "tmpl" },
    el("label", { text: `${k}: relief words ({style})` }, el("input", { id: `st-${k}-relief`, value: val.relief, disabled: locked })),
    el("label", { text: `${k}: detail sentence` }, el("textarea", { id: `st-${k}-detail`, rows: 2, disabled: locked }, val.detail)))));
  $("verAddText").replaceChildren(...Object.entries(t.additions).map(([k, val]) => templateBox(`ad-${k}`, k, 2, val, locked)));
  $("verExamples").replaceChildren(...Object.entries(v.examples).map(([cat, p]) =>
    el("div", { class: "example" }, el("b", { text: catLabel(cat) }), el("pre", { text: p }))));
  document.querySelectorAll("#versionList .ver").forEach((d) => d.classList.toggle("on", d.querySelector("b").textContent === v.version));
}

function collectVersion() {
  const t = S.version.templates;
  const templates = {};
  for (const [k] of TEMPLATE_FIELDS) templates[k] = $(`tpl-${k}`).value;
  templates.image_types = Object.fromEntries(Object.keys(t.image_types).map((k) => [k, $(`it-${k}`).value]));
  templates.styles = Object.fromEntries(Object.keys(t.styles).map((k) => [k, { relief: $(`st-${k}-relief`).value, detail: $(`st-${k}-detail`).value }]));
  templates.additions = Object.fromEntries(Object.keys(t.additions).map((k) => [k, $(`ad-${k}`).value]));
  return {
    notes: $("verNotes").value,
    settings: {
      model: $("verModel").value || null,
      style: $("verStyle").value,
      additions: [...document.querySelectorAll("#verAdditions .chip.on")].map((b) => b.dataset.id),
      fidelity: parseFloat($("verFidelity").value),
    },
    templates,
  };
}

async function saveVersion() {
  try {
    const v = await api(`/api/lab/prompts/${S.version.version}`, { method: "PUT", json: collectVersion() });
    S.version = null;
    await loadVersions(v.version);
    setStatus("verStatus", "Saved.");
  } catch (err) { setStatus("verStatus", err.message, true); }
}

async function newVersion() {
  const notes = prompt(`New version from ${S.version.version}. What will you change? (you can edit this later)`, "");
  if (notes === null) return;
  try {
    const v = await api("/api/lab/prompts", { method: "POST", json: { based_on: S.version.version, notes } });
    await loadVersions(v.version);
    setStatus("verStatus", `Created ${v.version}.`);
  } catch (err) { setStatus("verStatus", err.message, true); }
}

// --- runs ------------------------------------------------------------------------------

function runSelection() {
  const split = $("runSplit").value;
  const cats = [...document.querySelectorAll("#runCats .chip.on")].map((b) => b.dataset.id);
  return S.sources.filter((s) => (split === "all" || s.split === split) && (!cats.length || cats.includes(s.category)));
}

function versionPrice(name) {
  const v = S.versions.find((x) => x.version === name);
  const m = v && S.opts.models.find((x) => x.id === (v.settings && v.settings.model));
  return m ? m.price_usd : S.opts.provider === "mock" ? 0 : null;
}

function renderRunEstimate() {
  if (!S.opts) return;
  const n = runSelection().length;
  const price = versionPrice($("runVersion").value);
  $("runEstimate").textContent = `${n} image${n === 1 ? "" : "s"}` + (price == null ? "" : ` · about ${money(n * price)}`) +
    (S.opts.provider === "mock" ? " (offline mock provider: free, not a real model)" : "");
  $("runStart").disabled = n === 0;
}

async function startRun() {
  const n = runSelection().length;
  const price = versionPrice($("runVersion").value);
  const cost = price == null ? "" : ` (about ${money(n * price)})`;
  if (!confirm(`Generate ${n} image${n === 1 ? "" : "s"} with ${$("runVersion").value}${cost}?`)) return;
  try {
    const run = await api("/api/lab/runs", { method: "POST", json: {
      version: $("runVersion").value, split: $("runSplit").value,
      categories: [...document.querySelectorAll("#runCats .chip.on")].map((b) => b.dataset.id),
      notes: $("runNotes").value,
    } });
    setStatus("runStatus", `Started ${run.run_id}.`);
    $("runNotes").value = "";
    await loadRuns();
    await openRun(run.run_id);
  } catch (err) { setStatus("runStatus", err.message, true); }
}

async function loadRuns() {
  S.runs = (await api("/api/lab/runs")).runs;
  const body = $("runTable").querySelector("tbody");
  body.replaceChildren();
  if (!S.runs.length) body.append(el("tr", {}, el("td", { colspan: 8, class: "muted", text: "No runs yet." })));
  for (const r of S.runs) {
    const v = r.verdicts || {};
    body.append(el("tr", { class: `clickable${S.run && S.run.run_id === r.run_id ? " on" : ""}`, onclick: () => openRun(r.run_id) },
      el("td", { text: r.run_id + (r.notes ? ` · ${r.notes}` : "") }),
      el("td", { text: r.version }),
      el("td", { text: `${r.done}/${r.total}${r.failed ? ` (${r.failed} failed)` : ""}${r.active ? " · running" : ""}` }),
      el("td", { text: `${r.reviewed}/${r.done}` }),
      el("td", { text: `${v.good || 0} / ${v.usable || 0} / ${v.bad || 0}` }),
      el("td", { text: r.avg_score ?? "—" }),
      el("td", { text: money(r.usd) }),
      el("td", {}, el("button", { class: "ghost", text: "Review →" }))));
  }
  const cmp = $("revCompare"), keep = cmp.value;
  cmp.replaceChildren(el("option", { value: "", text: "—" }),
    ...S.runs.filter((r) => !S.run || r.run_id !== S.run.run_id).map((r) => el("option", { value: r.run_id, text: `${r.run_id} (${r.version})` })));
  cmp.value = [...cmp.options].some((o) => o.value === keep) ? keep : "";
}

async function openRun(runId, keepItem = false) {
  const prevId = keepItem && currentItem() ? currentItem().source_id : null;
  S.run = await api(`/api/lab/runs/${runId}`);
  $("review").hidden = false;
  const cats = [...new Set(S.run.items.map((i) => i.category))];
  const revCat = $("revCat"), keepCat = revCat.value;
  revCat.replaceChildren(el("option", { value: "", text: "All categories" }), ...cats.map((c) => el("option", { value: c, text: catLabel(c) })));
  revCat.value = cats.includes(keepCat) ? keepCat : "";
  renderRunHeader();
  renderStrip();
  const items = visibleItems();
  const idx = prevId ? items.findIndex((i) => i.source_id === prevId) : -1;
  if (idx >= 0) S.itemIdx = idx;
  else if (!keepItem) S.itemIdx = Math.max(0, items.findIndex((i) => i.status === "done" && !i.review));
  if (!keepItem || idx < 0) renderItem();
  document.querySelectorAll("#runTable tbody tr").forEach((tr) => tr.classList.toggle("on", tr.firstChild.textContent.startsWith(runId)));
  if (!keepItem) await loadRuns();
  schedulePoll();
}

function renderRunHeader() {
  const r = S.run, s = r.summary;
  $("revTitle").textContent = `${r.run_id} · ${r.version}${r.notes ? ` · ${r.notes}` : ""}`;
  $("revProgress").style.width = `${s.total ? (100 * (s.done + s.failed)) / s.total : 0}%`;
  const v = s.verdicts;
  const perCat = Object.entries(s.by_category).map(([c, b]) =>
    `${catLabel(c)}: ${b.verdicts.good}/${b.verdicts.usable}/${b.verdicts.bad}${b.avg_score ? ` (${b.avg_score})` : ""}`).join(" · ");
  const topErrors = Object.entries(s.errors).slice(0, 5).map(([t, n]) => `${tagLabel(t)} ×${n}`).join(", ");
  $("revSummary").textContent = `${s.done}/${s.total} made${s.failed ? `, ${s.failed} failed` : ""}${r.active ? " (running…)" : ""} · ` +
    `${s.reviewed} reviewed · good/usable/bad ${v.good}/${v.usable}/${v.bad}` + (s.avg_score ? ` · avg ${s.avg_score}` : "") +
    ` · ${money(s.usd)}` + (perCat ? `\n${perCat}` : "") + (topErrors ? `\nTop errors: ${topErrors}` : "");
  $("revSummary").style.whiteSpace = "pre-line";
  $("revStop").hidden = !r.active;
  $("revResume").hidden = r.active || s.done === s.total;
}

function visibleItems() {
  if (!S.run) return [];
  const f = $("revFilter").value, cat = $("revCat").value;
  return S.run.items.filter((i) => {
    if (cat && i.category !== cat) return false;
    if (!f) return true;
    if (f === "failed") return i.status === "failed";
    if (f === "unreviewed") return i.status === "done" && !(i.review && i.review.verdict);
    return i.review && i.review.verdict === f;
  });
}

const currentItem = () => visibleItems()[S.itemIdx];

function renderStrip() {
  const strip = $("revStrip");
  strip.replaceChildren();
  visibleItems().forEach((i, idx) => {
    const cls = ["thumb", idx === S.itemIdx ? "on" : "", i.status === "failed" ? "failed" : "", i.review && i.review.verdict ? i.review.verdict : ""].join(" ");
    strip.append(el("div", { class: cls, title: `${i.source_id} · ${i.status}`, onclick: () => { S.itemIdx = idx; renderItem(); renderStrip(); } },
      i.status === "done" ? el("img", { src: fileUrl(i.generation, 160), loading: "lazy" }) : el("div", { class: "pending", text: i.status }),
      el("div", { class: "lbl", text: i.source_id })));
  });
  const on = strip.querySelector(".thumb.on");
  if (on) on.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function figure(caption, path, status, error) {
  if (!path || status !== "done") {
    return el("figure", {}, el("figcaption", { text: caption }),
      el("div", { class: `placeholder${status === "failed" ? " error" : ""}`, text: status === "failed" ? `Failed: ${error || ""}` : status || "—" }));
  }
  return el("figure", {}, el("figcaption", { text: caption }),
    el("img", { src: fileUrl(path, 900), onclick: () => openLightbox(fileUrl(path)), alt: caption }));
}

function renderItem() {
  const i = currentItem();
  $("revItem").hidden = !i;
  if (!i) return;
  $("revItemTitle").textContent = `${i.source_id} · ${catLabel(i.category)} · ${i.split || ""}` +
    (i.seconds ? ` · ${i.seconds}s` : "") + (i.usd != null ? ` · ${money(i.usd)}` : "");
  $("revImages").replaceChildren(
    i.source ? el("figure", {}, el("figcaption", { text: "Source" }),
      el("img", { src: fileUrl(i.source, 900), onclick: () => openLightbox(fileUrl(i.source)), alt: "Source" })) : figure("Source", null, "missing"),
    figure(`Generation (${S.run.version})`, i.generation, i.status, i.error),
    figure("Relief preview (shaded depth)", i.relief, i.status, i.error),
    figure("Depth map (white = raised)", i.depth, i.status, i.error));
  renderCompare(i);
  renderReviewForm(i);
  $("revPrompt").textContent = i.prompt || "";
}

async function renderCompare(i) {
  const row = $("revCompareRow");
  const id = $("revCompare").value;
  if (!id) { row.hidden = true; S.compare = null; return; }
  if (!S.compare || S.compare.run_id !== id) S.compare = await api(`/api/lab/runs/${id}`);
  const c = S.compare.items.find((x) => x.source_id === i.source_id);
  row.hidden = false;
  if (!c) {
    row.replaceChildren(el("div", { class: "muted", text: `${i.source_id} isn't in ${id}.` }));
    return;
  }
  const r = c.review || {};
  const note = [r.verdict, r.score && `score ${r.score}`, (r.errors || []).map(tagLabel).join(", "), r.comment].filter(Boolean).join(" · ");
  row.replaceChildren(
    el("figure", {}, el("figcaption", { text: `${S.compare.version} review` }), el("div", { class: "placeholder", text: note || "not reviewed" })),
    figure(`Generation (${S.compare.version})`, c.generation, c.status, c.error),
    figure(`Relief preview (${S.compare.version})`, c.relief, c.status, c.error),
    figure(`Depth map (${S.compare.version})`, c.depth, c.status, c.error));
}

// --- review form -------------------------------------------------------------------------

function reviewOf(i) {
  return { verdict: null, score: null, errors: [], comment: "", ...(i.review || {}) };
}

function renderReviewForm(i) {
  const r = reviewOf(i);
  const enabled = i.status === "done";
  document.querySelectorAll("#revVerdict button").forEach((b) => { b.classList.toggle("on", b.dataset.v === r.verdict); b.disabled = !enabled; });
  document.querySelectorAll("#revScore button").forEach((b) => { b.classList.toggle("on", +b.dataset.s === r.score); b.disabled = !enabled; });
  const box = $("revTags");
  box.replaceChildren();
  const groupSel = $("revNewTagGroup"), keepGroup = groupSel.value;
  groupSel.replaceChildren();
  for (const g of S.opts.tags) {
    groupSel.append(el("option", { value: g.group, text: g.group }));
    box.append(el("div", { class: "tag-group" }, el("span", { class: "gname", text: g.group }),
      el("div", { class: "chips" }, ...g.tags.map((t) => el("button", {
        class: `chip${r.errors.includes(t) ? " on" : ""}`, disabled: !enabled, text: tagLabel(t),
        onclick: () => toggleTag(t),
      })))));
  }
  if (keepGroup) groupSel.value = keepGroup;
  const c = $("revComment");
  if (document.activeElement !== c) c.value = r.comment || "";
  c.disabled = !enabled;
  $("revSaved").textContent = i.review && i.review.reviewed_at ? `saved ${i.review.reviewed_at.replace("T", " ")}` : "";
}

// Saves go one at a time per result, always sending its latest state, and a
// reply never overwrites what has been typed since (quick keys raced before).
const saves = new Map(); // "run/source" -> { timer, inflight, dirty }

function updateReview(patch, immediate = true) {
  const i = currentItem();
  if (!i || i.status !== "done") return;
  i.review = { ...reviewOf(i), ...patch };
  renderReviewForm(i);
  queueSave(S.run.run_id, i, immediate ? 0 : 700);
}

function queueSave(run, item, delay) {
  const key = `${run}/${item.source_id}`;
  const st = saves.get(key) || { timer: 0, inflight: false, dirty: false };
  saves.set(key, st);
  clearTimeout(st.timer);
  $("revSaved").textContent = "saving…";
  st.timer = setTimeout(() => sendSave(run, item, st), delay);
}

async function sendSave(run, item, st) {
  if (st.inflight) { st.dirty = true; return; }
  st.inflight = true;
  st.dirty = false;
  const body = reviewOf(item);
  try {
    const out = await api(`/api/lab/runs/${run}/items/${item.source_id}/review`, { method: "PUT", json: {
      verdict: body.verdict, score: body.score, errors: body.errors, comment: body.comment,
    } });
    if (out.review && item.review) item.review.reviewed_at = out.review.reviewed_at;
    if (!out.review) item.review = null;
    if (currentItem() === item && !st.dirty) $("revSaved").textContent = out.review ? "saved" : "cleared";
    renderStrip();
    refreshSummary();
  } catch (err) {
    if (currentItem() === item) $("revSaved").textContent = `not saved: ${err.message}`;
  } finally {
    st.inflight = false;
    if (st.dirty) sendSave(run, item, st);
  }
}

function toggleTag(t) {
  const r = reviewOf(currentItem());
  updateReview({ errors: r.errors.includes(t) ? r.errors.filter((x) => x !== t) : [...r.errors, t] });
}

async function refreshSummary() {
  if (!S.run) return;
  const fresh = await api(`/api/lab/runs/${S.run.run_id}`);
  S.run.summary = fresh.summary;
  S.run.active = fresh.active;
  renderRunHeader();
}

async function addTag() {
  const label = $("revNewTag").value.trim();
  if (!label) return;
  try {
    const out = await api("/api/lab/tags", { method: "POST", json: { group: $("revNewTagGroup").value, label } });
    S.opts.tags = out.tags;
    $("revNewTag").value = "";
    const r = reviewOf(currentItem());
    if (!r.errors.includes(out.tag)) updateReview({ errors: [...r.errors, out.tag] });
    else renderReviewForm(currentItem());
  } catch (err) { $("revSaved").textContent = err.message; }
}

function move(delta) {
  const n = visibleItems().length;
  if (!n) return;
  S.itemIdx = Math.min(n - 1, Math.max(0, S.itemIdx + delta));
  renderItem();
  renderStrip();
}

function nextUnreviewed() {
  const items = visibleItems();
  const n = items.length;
  for (let k = 1; k <= n; k++) {
    const idx = (S.itemIdx + k) % n;
    if (items[idx].status === "done" && !(items[idx].review && items[idx].review.verdict)) {
      S.itemIdx = idx;
      renderItem();
      renderStrip();
      return;
    }
  }
}

function onKey(e) {
  if ($("tab-runs").hidden || !S.run || $("revItem").hidden) return;
  if (e.key === "Escape") {
    if (!$("lightbox").hidden) { $("lightbox").hidden = true; return; }
    if (document.activeElement) document.activeElement.blur();
    return;
  }
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "arrowright" || k === "j") move(1);
  else if (k === "arrowleft" || k === "k") move(-1);
  else if (k === "g") updateReview({ verdict: "good" });
  else if (k === "u") updateReview({ verdict: "usable" });
  else if (k === "b") updateReview({ verdict: "bad" });
  else if (k === "n") nextUnreviewed();
  else if (/^[1-5]$/.test(k)) updateReview({ score: +k });
  else return;
  e.preventDefault();
}

// --- polling while a run is going ------------------------------------------------------------

function schedulePoll() {
  clearTimeout(S.pollTimer);
  if (!S.run || !S.run.active) return;
  S.pollTimer = setTimeout(async () => {
    if (!S.run) return;
    const before = currentItem();
    const beforeStatus = before && before.status;
    const fresh = await api(`/api/lab/runs/${S.run.run_id}`);
    // keep reviews typed since the last save
    for (const it of fresh.items) {
      const old = S.run.items.find((x) => x.source_id === it.source_id);
      if (old && old.review && !it.review) it.review = old.review;
    }
    S.run = fresh;
    renderRunHeader();
    renderStrip();
    const now = currentItem();
    if (!before || (now && now.status !== beforeStatus)) renderItem();
    if (!fresh.active) loadRuns();
    schedulePoll();
  }, 2000);
}

function wireRuns() {
  for (const c of S.opts.categories) {
    $("runCats").append(el("button", { class: "chip", dataset: { id: c.id }, onclick: (e) => { e.currentTarget.classList.toggle("on"); renderRunEstimate(); } }, c.label));
  }
  $("runVersion").addEventListener("change", renderRunEstimate);
  $("runSplit").addEventListener("change", renderRunEstimate);
  $("runStart").addEventListener("click", startRun);
  $("revFilter").addEventListener("change", () => { S.itemIdx = 0; renderStrip(); renderItem(); });
  $("revCat").addEventListener("change", () => { S.itemIdx = 0; renderStrip(); renderItem(); });
  $("revCompare").addEventListener("change", () => { const i = currentItem(); if (i) renderCompare(i); });
  $("revStop").addEventListener("click", async () => { await api(`/api/lab/runs/${S.run.run_id}/stop`, { method: "POST" }); $("revStop").disabled = true; setTimeout(() => { $("revStop").disabled = false; }, 3000); });
  $("revResume").addEventListener("click", async () => {
    try { await api(`/api/lab/runs/${S.run.run_id}/resume`, { method: "POST", json: {} }); await openRun(S.run.run_id, true); }
    catch (err) { setStatus("runStatus", err.message, true); }
  });
  document.querySelectorAll("#revVerdict button").forEach((b) => b.addEventListener("click", () => {
    const r = reviewOf(currentItem());
    updateReview({ verdict: r.verdict === b.dataset.v ? null : b.dataset.v });
  }));
  document.querySelectorAll("#revScore button").forEach((b) => b.addEventListener("click", () => {
    const r = reviewOf(currentItem());
    updateReview({ score: r.score === +b.dataset.s ? null : +b.dataset.s });
  }));
  $("revComment").addEventListener("input", (e) => updateReview({ comment: e.target.value }, false));
  $("revAddTag").addEventListener("click", addTag);
  $("revNewTag").addEventListener("keydown", (e) => { if (e.key === "Enter") addTag(); });
  document.addEventListener("keydown", onKey);
}

// --- start ------------------------------------------------------------------------------------

async function init() {
  S.opts = await api("/api/lab/options");
  $("labDir").textContent = S.opts.lab_dir;
  $("labDir").title = "Where the lab's files are kept (PROMPT_LAB_DIR)";
  wireImages();
  wireRuns();
  $("verSave").addEventListener("click", saveVersion);
  $("verNew").addEventListener("click", newVersion);
  await loadSources();
  await loadVersions();
  let tab = "images";
  try { tab = localStorage.getItem("lab.tab") || "images"; } catch { /* storage unavailable */ }
  showTab(tab);
}

init().catch((err) => {
  document.body.prepend(el("div", { class: "status error", text: `Couldn't load the prompt lab: ${err.message}` }));
});
