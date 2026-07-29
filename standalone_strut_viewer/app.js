"use strict";

const state = {
  catalog: null,
  selectedEntry: null,
  profile: null,
  volume: null,
  profileCache: new Map(),
  volumeCache: new Map(),
  profileAbort: null,
  trackedOverlay: true,
  registeredOverlay: true,
};

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) =>
  value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);

const inputs = {
  tiff: $("tiffFile"),
  json: $("jsonFile"),
  analyses: $("analysisFiles"),
};

function setStatus(message, kind = "") {
  const target = $("loadStatus");
  target.textContent = message;
  target.className = `load-status ${kind}`.trim();
}

function selectedFile(input) {
  return input.files && input.files.length ? input.files[0] : null;
}

function refreshFileState() {
  $("tiffName").textContent = selectedFile(inputs.tiff)?.name || "Choose a 3D .tif or .tiff";
  $("jsonName").textContent = selectedFile(inputs.json)?.name || "Junction and strut voxel positions";
  const analyses = [...(inputs.analyses.files || [])];
  $("analysisName").textContent = analyses.length
    ? `${analyses.length} JSON file${analyses.length === 1 ? "" : "s"} selected`
    : "Select one or more finding / hand-off JSONs";
  const ready = selectedFile(inputs.tiff) && selectedFile(inputs.json) && analyses.length;
  $("loadFiles").disabled = !ready;
  if (ready) setStatus(`Ready to load ${analyses.length + 2} files.`);
  else setStatus("Select a TIFF, registered JSON, and analysis JSON files.");
}

async function uploadFile(url, file) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "X-File-Name": encodeURIComponent(file.name) },
    body: file,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status}).`);
  return payload;
}

async function loadFiles() {
  const tiff = selectedFile(inputs.tiff);
  const json = selectedFile(inputs.json);
  const analyses = [...(inputs.analyses.files || [])];
  if (!tiff || !json || !analyses.length) return;
  const button = $("loadFiles");
  button.disabled = true;
  try {
    await fetch("/api/session", { method: "DELETE" });
    setStatus("Streaming TIFF to temporary local storage…");
    const tiffResult = await uploadFile("/api/tiff", tiff);
    setStatus(`TIFF ready (${tiffResult.shape_zyx.join(" × ")} voxels). Reading registration…`);
    await uploadFile("/api/registration", json);
    setStatus(`Registration ready. Reading ${analyses.length} analysis JSON files...`);
    await Promise.all(analyses.map((file) => uploadFile("/api/results", file)));
    const response = await fetch("/api/catalog", { cache: "no-store" });
    const catalog = await response.json();
    if (!response.ok) throw new Error(catalog.error || "Could not build the strut list.");
    if (!catalog.entries.length) {
      throw new Error("No analysis JSON strut IDs were found in the registered JSON.");
    }
    state.catalog = catalog;
    state.selectedEntry = null;
    state.profile = null;
    state.profileCache.clear();
    state.volumeCache.clear();
    renderCatalog();
    setStatus(
      `Loaded ${catalog.entries.length.toLocaleString()} struts from ${analyses.length} JSON files. CT threshold: ${fmt(catalog.threshold, 1)}.`,
      "success"
    );
    $("clearFiles").hidden = false;
  } catch (error) {
    setStatus(error.message || String(error), "error");
    button.disabled = false;
  }
}

async function clearFiles() {
  await fetch("/api/session", { method: "DELETE" }).catch(() => {});
  Object.values(inputs).forEach((input) => { input.value = ""; });
  state.catalog = state.selectedEntry = state.profile = state.volume = null;
  state.profileCache.clear();
  state.volumeCache.clear();
  $("workspace").hidden = true;
  $("clearFiles").hidden = true;
  $("strutDetail").hidden = true;
  $("emptyState").hidden = false;
  refreshFileState();
}

function renderCatalog() {
  $("workspace").hidden = false;
  $("strutCount").textContent = `${state.catalog.entries.length.toLocaleString()} struts`;
  const missing = state.catalog.unmatched_ids || [];
  $("unmatchedNote").textContent = missing.length
    ? `${missing.length} analysis ID${missing.length === 1 ? " was" : "s were"} not present in the registered JSON.`
    : "";
  $("searchInput").value = "";
  const filter = $("defectFilter");
  filter.innerHTML = '<option value="all">All classifications</option>';
  Object.entries(state.catalog.class_counts || {}).forEach(([name, count]) => {
    if (!count) return;
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name[0].toUpperCase()}${name.slice(1)} (${count})`;
    filter.appendChild(option);
  });
  renderStrutList();
}

function renderStrutList() {
  const query = $("searchInput").value.trim().toLowerCase();
  const defect = $("defectFilter").value;
  const values = state.catalog.entries
    .filter((entry) =>
      (!query || String(entry.strut_id).toLowerCase().includes(query)) &&
      (defect === "all" || entry.classifications.includes(defect))
    )
    .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0));
  const list = $("strutList");
  list.innerHTML = "";
  values.slice(0, 250).forEach((entry) => {
    const button = document.createElement("button");
    button.className = `strut-row ${state.selectedEntry?.strut_id === entry.strut_id ? "active" : ""}`;
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", state.selectedEntry?.strut_id === entry.strut_id ? "true" : "false");
    button.innerHTML =
      `<b>#${entry.strut_id}</b>` +
      `<span class="class-badge ${entry.classification}">${entry.classifications.join(" + ")}</span>` +
      `<small>${fmt(entry.confidence, 2)} confidence</small>`;
    button.addEventListener("click", () => selectStrut(entry));
    list.appendChild(button);
  });
  if (!values.length) {
    list.innerHTML = '<p class="unmatched-note">No struts match the current search and filter.</p>';
  } else if (values.length > 250) {
    const note = document.createElement("p");
    note.className = "unmatched-note";
    note.textContent = `Showing the first 250 of ${values.length.toLocaleString()} matches. Search by ID to narrow the list.`;
    list.appendChild(note);
  }
}

async function selectStrut(entry) {
  state.selectedEntry = entry;
  state.profile = null;
  renderStrutList();
  $("emptyState").hidden = true;
  $("strutDetail").hidden = false;
  $("detailId").textContent = `#${entry.strut_id}`;
  $("coverageMetric").textContent = "Loading...";
  $("radiusMetric").textContent = "—";
  $("deviationMetric").textContent = "—";
  $("classificationMetric").textContent = entry.classifications.join(" + ");
  $("confidenceMetric").textContent = fmt(entry.confidence, 2);
  $("thresholdMetric").textContent = fmt(state.catalog.threshold, 1);
  $("profileSource").textContent = entry.has_embedded_measurements
    ? "Loading embedded pipeline measurements..."
    : "No embedded profile—loading a viewer preview.";
  $("profileSource").className = `profile-source ${
    entry.has_embedded_measurements ? "pipeline" : "preview"
  }`;
  renderMetadata({
    classifications: entry.classifications.join(", "),
    sources: entry.sources.join(", "),
    registered_length_voxels: entry.length_voxels,
    ...entry.fields,
  });
  clearGraphs();
  if (state.profileAbort) state.profileAbort.abort();
  const cacheKey = `${entry.strut_id}:${state.catalog.threshold}`;
  try {
    let payload = state.profileCache.get(cacheKey);
    if (!payload) {
      state.profileAbort = new AbortController();
      const response = await fetch(
        `/api/profile/${entry.strut_id}?threshold=${encodeURIComponent(state.catalog.threshold)}`,
        { cache: "no-store", signal: state.profileAbort.signal }
      );
      payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Profile calculation failed.");
      state.profileCache.set(cacheKey, payload);
      if (state.profileCache.size > 128) {
        state.profileCache.delete(state.profileCache.keys().next().value);
      }
    }
    if (state.selectedEntry?.strut_id !== entry.strut_id) return;
    state.profile = payload;
    $("coverageMetric").textContent = `${fmt(100 * payload.coverage, 0)}%`;
    $("radiusMetric").textContent = `${fmt(payload.median_radius_voxels, 2)} vox`;
    $("deviationMetric").textContent = `${fmt(payload.centerline_deviation_max_voxels, 2)} vox`;
    if (Number.isFinite(Number(payload.threshold))) {
      $("thresholdMetric").textContent = fmt(payload.threshold, 1);
    }
    const embedded = payload.profile_source === "embedded_pipeline";
    $("profileSource").textContent = embedded
      ? `Pipeline measurements · threshold ${fmt(payload.threshold, 1)} · ${
          payload.section_measurements_sha256
            ? `sections ${payload.section_measurements_sha256.slice(0, 12)}…`
            : "verified profile"
        }`
      : "Viewer preview—not classification evidence";
    $("profileSource").className = `profile-source ${
      embedded ? "pipeline" : "preview"
    }`;
    drawRadiusGraph(payload.profile, entry.fields);
    drawDeviationGraph(payload.profile);
  } catch (error) {
    if (error.name === "AbortError") return;
    $("coverageMetric").textContent = "Unavailable";
    $("profileSource").textContent = "Measurement profile unavailable";
    $("profileSource").className = "profile-source preview";
    setStatus(error.message || String(error), "error");
  }
}

function renderMetadata(fields) {
  const target = $("metadataGrid");
  const entries = Object.entries(fields || {});
  target.innerHTML = "";
  if (!entries.length) {
    target.innerHTML = "<div><span>JSON metadata</span><b>No additional fields</b></div>";
    return;
  }
  entries.forEach(([key, value]) => {
    const item = document.createElement("div");
    const label = document.createElement("span");
    const content = document.createElement("b");
    label.textContent = key.replaceAll("_", " ");
    const display = typeof value === "object" ? JSON.stringify(value) : String(value);
    content.textContent = display;
    content.title = display;
    item.append(label, content);
    target.appendChild(item);
  });
}

function clearGraphs() {
  ["radiusGraph", "deviationGraph"].forEach((id) => {
    const canvas = $(id);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.round(300 * devicePixelRatio));
    const context = canvas.getContext("2d");
    context.scale(devicePixelRatio, devicePixelRatio);
    context.clearRect(0, 0, rect.width, 300);
  });
}

function drawSeriesGraph(canvasId, profile, field, color, ylabel, reference = null) {
  const canvas = $(canvasId);
  const width = canvas.clientWidth || 700;
  const height = 300;
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  const pad = { left: 52, right: 22, top: 24, bottom: 40 };
  const graphWidth = width - pad.left - pad.right;
  const graphHeight = height - pad.top - pad.bottom;
  const values = profile.map((item) => item[field]).filter(Number.isFinite);
  const maximum = Math.max(1, ...values, Number(reference || 0)) * 1.16;

  context.strokeStyle = "#dbe4ec";
  context.lineWidth = 1;
  for (let step = 0; step <= 4; step += 1) {
    const y = pad.top + step * graphHeight / 4;
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.fillStyle = "#789";
    context.font = "10px system-ui";
    context.fillText(fmt(maximum * (1 - step / 4), 1), 15, y + 3);
  }

  if (Number.isFinite(reference)) {
    const y = pad.top + (1 - reference / maximum) * graphHeight;
    context.save();
    context.setLineDash([7, 5]);
    context.strokeStyle = "#d08b2e";
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.restore();
  }
  context.strokeStyle = color;
  context.lineWidth = 2.5;
  context.beginPath();
  let started = false;
  profile.forEach((item) => {
    if (!Number.isFinite(item[field])) {
      started = false;
      return;
    }
    const x = pad.left + item.fraction * graphWidth;
    const y = pad.top + (1 - item[field] / maximum) * graphHeight;
    if (started) context.lineTo(x, y);
    else {
      context.moveTo(x, y);
      started = true;
    }
  });
  context.stroke();

  profile.forEach((item) => {
    if (!Number.isFinite(item[field])) return;
    const x = pad.left + item.fraction * graphWidth;
    const y = pad.top + (1 - item[field] / maximum) * graphHeight;
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, 3.3, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = "#64748b";
  context.font = "11px system-ui";
  context.fillText("position along strut", width / 2 - 48, height - 10);
  context.save();
  context.translate(10, height / 2 + 30);
  context.rotate(-Math.PI / 2);
  context.fillText(ylabel, 0, 0);
  context.restore();
}

function drawRadiusGraph(profile, fields) {
  const peerRadius = Number(fields.peer_median_radius_voxels);
  drawSeriesGraph(
    "radiusGraph", profile, "radius_voxels", "#087bbd", "radius (voxels)",
    Number.isFinite(peerRadius) ? peerRadius : null
  );
}

function drawDeviationGraph(profile) {
  const threshold = Number(state.catalog.thresholds?.bent_adjacent_deviation_voxels);
  drawSeriesGraph(
    "deviationGraph", profile, "deviation_voxels", "#8b4db5",
    "deviation (voxels)", Number.isFinite(threshold) ? threshold : null
  );
}

function parseHeaderNumbers(response, name) {
  return response.headers.get(name).split(",").map(Number);
}

async function openFourViews() {
  if (!state.selectedEntry || !state.profile) return;
  const button = $("openViews");
  button.disabled = true;
  button.textContent = "Loading volume...";
  try {
    const cacheKey = state.selectedEntry.strut_id;
    state.volume = state.volumeCache.get(cacheKey);
    if (!state.volume) {
      const response = await fetch(`/api/volume/${cacheKey}`, { cache: "no-store" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "Could not load the local CT crop.");
      }
      const buffer = await response.arrayBuffer();
      const dtype = response.headers.get("X-Volume-Dtype");
      const constructors = {
        uint16: Uint16Array,
        int16: Int16Array,
        float32: Float32Array,
      };
      const TypedArray = constructors[dtype] || Float32Array;
      state.volume = {
        data: new TypedArray(buffer),
        shape: parseHeaderNumbers(response, "X-Volume-Shape"),
        origin: parseHeaderNumbers(response, "X-Volume-Origin-XYZ"),
        start: parseHeaderNumbers(response, "X-Strut-Start-XYZ"),
        end: parseHeaderNumbers(response, "X-Strut-End-XYZ"),
        range: parseHeaderNumbers(response, "X-Intensity-Range"),
      };
      state.volumeCache.set(cacheKey, state.volume);
      if (state.volumeCache.size > 2) {
        state.volumeCache.delete(state.volumeCache.keys().next().value);
      }
    }
    state.trackedOverlay = true;
    state.registeredOverlay = true;
    updateOverlayButton("overlayToggle", "CT tracking", true);
    updateOverlayButton("registeredToggle", "Registered position", true);
    $("modalTitle").textContent = `Strut #${state.selectedEntry.strut_id}`;
    configureViewSliders();
    $("viewerModal").hidden = false;
    document.body.style.overflow = "hidden";
    drawAllViews();
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = "Open four views";
  }
}

function configureViewSliders() {
  const { shape, origin, start, end } = state.volume;
  const trackedMidpoint = interpolateProfile(0.5);
  const midpoint = trackedMidpoint
    ? trackedPoint(trackedMidpoint)
    : start.map((value, index) => 0.5 * (value + end[index]));
  const settings = [
    [$("xySlider"), shape[0], Math.round(midpoint[2] - origin[2])],
    [$("xzSlider"), shape[1], Math.round(midpoint[1] - origin[1])],
    [$("yzSlider"), shape[2], Math.round(midpoint[0] - origin[0])],
  ];
  settings.forEach(([slider, size, value]) => {
    slider.min = "0";
    slider.max = String(Math.max(0, size - 1));
    slider.value = String(Math.max(0, Math.min(size - 1, value)));
  });
  $("perpSlider").value = "50";
}

function volumeIndex(z, y, x) {
  const [, ny, nx] = state.volume.shape;
  return (z * ny + y) * nx + x;
}

function gray(value) {
  const [low, high] = state.volume.range;
  return Math.max(0, Math.min(255, Math.round(255 * (value - low) / Math.max(high - low, 1e-6))));
}

function vectorAdd(a, b) { return a.map((value, index) => value + b[index]); }
function vectorSub(a, b) { return a.map((value, index) => value - b[index]); }
function vectorScale(a, scale) { return a.map((value) => value * scale); }
function vectorDot(a, b) { return a.reduce((sum, value, index) => sum + value * b[index], 0); }
function vectorCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function vectorUnit(a) {
  const length = Math.sqrt(vectorDot(a, a)) || 1;
  return a.map((value) => value / length);
}

function strutBasis() {
  const direction = vectorUnit(vectorSub(state.volume.end, state.volume.start));
  let helper = [0, 0, 1];
  if (Math.abs(vectorDot(direction, helper)) > 0.9) helper = [0, 1, 0];
  const u = vectorUnit(vectorCross(direction, helper));
  const v = vectorUnit(vectorCross(direction, u));
  return { direction, u, v };
}

function interpolateProfile(fraction) {
  const values = state.profile.profile;
  if (!values.length) return null;
  if (fraction <= values[0].fraction) return values[0];
  if (fraction >= values.at(-1).fraction) return values.at(-1);
  for (let index = 1; index < values.length; index += 1) {
    if (fraction <= values[index].fraction) {
      const a = values[index - 1];
      const b = values[index];
      const t = (fraction - a.fraction) / Math.max(b.fraction - a.fraction, 1e-6);
      const lerp = (key) =>
        Number.isFinite(a[key]) && Number.isFinite(b[key])
          ? a[key] + t * (b[key] - a[key])
          : Number.isFinite(a[key]) ? a[key] : b[key];
      return {
        fraction,
        center_u_voxels: lerp("center_u_voxels"),
        center_v_voxels: lerp("center_v_voxels"),
        radius_voxels: lerp("radius_voxels"),
        confidence: lerp("confidence"),
      };
    }
  }
  return values.at(-1);
}

function trackedPoint(profileItem) {
  const { u, v } = strutBasis();
  const delta = vectorSub(state.volume.end, state.volume.start);
  let point = vectorAdd(state.volume.start, vectorScale(delta, profileItem.fraction));
  if (Number.isFinite(profileItem.center_u_voxels)) {
    point = vectorAdd(point, vectorScale(u, profileItem.center_u_voxels));
  }
  if (Number.isFinite(profileItem.center_v_voxels)) {
    point = vectorAdd(point, vectorScale(v, profileItem.center_v_voxels));
  }
  return point;
}

function drawOverlayDot(context, x, y) {
  context.fillStyle = "rgba(36,201,232,.38)";
  context.beginPath();
  context.arc(x, y, 7, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "#24c9e8";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x - 8, y);
  context.lineTo(x + 8, y);
  context.moveTo(x, y - 8);
  context.lineTo(x, y + 8);
  context.stroke();
}

function drawRegisteredMarker(context, x, y) {
  context.fillStyle = "rgba(239,108,87,.24)";
  context.beginPath();
  context.arc(x, y, 7, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "#ef6c57";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(x, y, 5, 0, Math.PI * 2);
  context.moveTo(x - 9, y);
  context.lineTo(x + 9, y);
  context.moveTo(x, y - 9);
  context.lineTo(x, y + 9);
  context.stroke();
}

function registeredGeometryForSlice(axisIndex, globalSlice) {
  const { start, end } = state.volume;
  const delta = end[axisIndex] - start[axisIndex];
  if (Math.abs(delta) <= 1e-6) {
    return Math.abs(globalSlice - start[axisIndex]) <= 0.6
      ? { segment: [start, end] }
      : null;
  }
  const fraction = (globalSlice - start[axisIndex]) / delta;
  if (fraction < -1e-6 || fraction > 1 + 1e-6) return null;
  const clamped = Math.max(0, Math.min(1, fraction));
  return {
    point: start.map((value, index) =>
      value + clamped * (end[index] - value)
    ),
  };
}

function drawRegisteredGeometry(context, geometry, pointToCanvas) {
  if (!geometry) return;
  if (geometry.point) {
    const [x, y] = pointToCanvas(geometry.point);
    drawRegisteredMarker(context, x, y);
    return;
  }
  const [a, b] = geometry.segment.map(pointToCanvas);
  context.save();
  context.strokeStyle = "#ef6c57";
  context.lineWidth = 2;
  context.setLineDash([5, 4]);
  context.beginPath();
  context.moveTo(a[0], a[1]);
  context.lineTo(b[0], b[1]);
  context.stroke();
  context.restore();
  drawRegisteredMarker(context, a[0], a[1]);
  drawRegisteredMarker(context, b[0], b[1]);
}

function trackedPointForSlice(axisIndex, globalSlice) {
  const samples = state.profile.profile
    .filter((item) =>
      Number.isFinite(item.center_u_voxels) &&
      Number.isFinite(item.center_v_voxels)
    )
    .map((item) => ({ item, point: trackedPoint(item) }));
  if (!samples.length) return null;

  for (let index = 1; index < samples.length; index += 1) {
    const a = samples[index - 1];
    const b = samples[index];
    const aAxis = a.point[axisIndex];
    const bAxis = b.point[axisIndex];
    if ((globalSlice - aAxis) * (globalSlice - bAxis) <= 0 &&
        Math.abs(bAxis - aAxis) > 1e-6) {
      const t = (globalSlice - aAxis) / (bAxis - aAxis);
      return {
        point: a.point.map((value, coordinate) =>
          value + t * (b.point[coordinate] - value)
        ),
        radius: (
          Number.isFinite(a.item.radius_voxels) &&
          Number.isFinite(b.item.radius_voxels)
        )
          ? a.item.radius_voxels + t * (b.item.radius_voxels - a.item.radius_voxels)
          : null,
        distance: 0,
      };
    }
  }

  const gaps = samples.slice(1).map((sample, index) =>
    Math.abs(sample.point[axisIndex] - samples[index].point[axisIndex])
  );
  const tolerance = Math.max(1.5, (gaps.length ? Math.max(...gaps) : 0) / 2 + 0.75);
  let best = null;
  samples.forEach((sample) => {
    const distance = Math.abs(sample.point[axisIndex] - globalSlice);
    if (!best || distance < best.distance) {
      best = {
        point: sample.point,
        radius: sample.item.radius_voxels,
        distance,
      };
    }
  });
  return best && best.distance <= tolerance ? best : null;
}

function drawOrthogonal(kind) {
  const { shape, origin } = state.volume;
  const [nz, ny, nx] = shape;
  let width;
  let height;
  let slider;
  let canvas;
  let label;
  let axisIndex;
  let globalSlice;
  let valueAt;
  let pointToCanvas;
  if (kind === "xy") {
    width = nx; height = ny; slider = $("xySlider"); canvas = $("xyCanvas"); label = $("xyLabel");
    const z = Number(slider.value); axisIndex = 2; globalSlice = origin[2] + z;
    valueAt = (row, col) => state.volume.data[volumeIndex(z, row, col)];
    pointToCanvas = (point) => [point[0] - origin[0], point[1] - origin[1]];
    label.textContent = `global Z ${globalSlice}`;
  } else if (kind === "xz") {
    width = nx; height = nz; slider = $("xzSlider"); canvas = $("xzCanvas"); label = $("xzLabel");
    const y = Number(slider.value); axisIndex = 1; globalSlice = origin[1] + y;
    valueAt = (row, col) => state.volume.data[volumeIndex(row, y, col)];
    pointToCanvas = (point) => [point[0] - origin[0], point[2] - origin[2]];
    label.textContent = `global Y ${globalSlice}`;
  } else {
    width = ny; height = nz; slider = $("yzSlider"); canvas = $("yzCanvas"); label = $("yzLabel");
    const x = Number(slider.value); axisIndex = 0; globalSlice = origin[0] + x;
    valueAt = (row, col) => state.volume.data[volumeIndex(row, col, x)];
    pointToCanvas = (point) => [point[1] - origin[1], point[2] - origin[2]];
    label.textContent = `global X ${globalSlice}`;
  }
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const intensity = gray(valueAt(row, col));
      const index = 4 * (row * width + col);
      image.data[index] = image.data[index + 1] = image.data[index + 2] = intensity;
      image.data[index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  if (state.trackedOverlay) {
    const tracked = trackedPointForSlice(axisIndex, globalSlice);
    if (tracked) {
      const [x, y] = pointToCanvas(tracked.point);
      drawOverlayDot(context, x, y);
    }
  }
  if (state.registeredOverlay) {
    drawRegisteredGeometry(
      context,
      registeredGeometryForSlice(axisIndex, globalSlice),
      pointToCanvas
    );
  }
}

function sampleVolume(point) {
  const { data, shape, origin } = state.volume;
  const [nz, ny, nx] = shape;
  const x = point[0] - origin[0];
  const y = point[1] - origin[1];
  const z = point[2] - origin[2];
  if (x < 0 || y < 0 || z < 0 || x > nx - 1 || y > ny - 1 || z > nz - 1) return state.volume.range[0];
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, nx - 1), y1 = Math.min(y0 + 1, ny - 1), z1 = Math.min(z0 + 1, nz - 1);
  const tx = x - x0, ty = y - y0, tz = z - z0;
  const at = (zz, yy, xx) => data[(zz * ny + yy) * nx + xx];
  const c00 = at(z0, y0, x0) * (1 - tx) + at(z0, y0, x1) * tx;
  const c01 = at(z0, y1, x0) * (1 - tx) + at(z0, y1, x1) * tx;
  const c10 = at(z1, y0, x0) * (1 - tx) + at(z1, y0, x1) * tx;
  const c11 = at(z1, y1, x0) * (1 - tx) + at(z1, y1, x1) * tx;
  const c0 = c00 * (1 - ty) + c01 * ty;
  const c1 = c10 * (1 - ty) + c11 * ty;
  return c0 * (1 - tz) + c1 * tz;
}

function nearestComponent(mask, size, center, maximumDistancePixels) {
  const visited = new Uint8Array(mask.length);
  let best = null;
  const neighbors = [-1, 0, 1];
  for (let seed = 0; seed < mask.length; seed += 1) {
    if (!mask[seed] || visited[seed]) continue;
    const queue = [seed];
    visited[seed] = 1;
    const pixels = [];
    let nearest = Infinity;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const index = queue[cursor];
      pixels.push(index);
      const row = Math.floor(index / size);
      const col = index % size;
      nearest = Math.min(nearest, Math.hypot(col - center, row - center));
      neighbors.forEach((dy) => neighbors.forEach((dx) => {
        if (dx === 0 && dy === 0) return;
        const yy = row + dy, xx = col + dx;
        if (yy < 0 || xx < 0 || yy >= size || xx >= size) return;
        const next = yy * size + xx;
        if (mask[next] && !visited[next]) {
          visited[next] = 1;
          queue.push(next);
        }
      }));
    }
    if (pixels.length >= 3 && nearest <= maximumDistancePixels &&
        (!best || nearest < best.nearest || (nearest === best.nearest && pixels.length > best.pixels.length))) {
      best = { pixels, nearest };
    }
  }
  return best;
}

function drawPerpendicular() {
  const fraction = Number($("perpSlider").value) / 100;
  const profileItem = interpolateProfile(fraction);
  const { u, v } = strutBasis();
  const delta = vectorSub(state.volume.end, state.volume.start);
  const registeredCenter = vectorAdd(state.volume.start, vectorScale(delta, fraction));
  let center = registeredCenter;
  if (profileItem && Number.isFinite(profileItem.center_u_voxels)) {
    center = vectorAdd(center, vectorScale(u, profileItem.center_u_voxels));
  }
  if (profileItem && Number.isFinite(profileItem.center_v_voxels)) {
    center = vectorAdd(center, vectorScale(v, profileItem.center_v_voxels));
  }
  const canvas = $("perpCanvas");
  const size = 81;
  const extent = state.profile.extent_voxels || 18;
  const spacing = 2 * extent / (size - 1);
  const middle = (size - 1) / 2;
  const values = new Float32Array(size * size);
  canvas.width = canvas.height = size;
  const context = canvas.getContext("2d");
  const image = context.createImageData(size, size);
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const du = (col - middle) * spacing;
      const dv = (row - middle) * spacing;
      const point = vectorAdd(vectorAdd(center, vectorScale(u, du)), vectorScale(v, dv));
      const value = sampleVolume(point);
      const index = row * size + col;
      values[index] = value;
      const intensity = gray(value);
      image.data[4 * index] = image.data[4 * index + 1] = image.data[4 * index + 2] = intensity;
      image.data[4 * index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const mask = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    mask[index] = values[index] >= state.profile.threshold ? 1 : 0;
  }
  const component = nearestComponent(mask, size, middle, 12 / spacing);
  let radius = null;
  let area = null;
  if (component) {
    area = component.pixels.length * spacing * spacing;
    radius = Math.sqrt(area / Math.PI);
    if (state.trackedOverlay) {
      const overlay = context.getImageData(0, 0, size, size);
      component.pixels.forEach((index) => {
        overlay.data[4 * index] = 36;
        overlay.data[4 * index + 1] = 201;
        overlay.data[4 * index + 2] = 232;
        overlay.data[4 * index + 3] = 255;
      });
      context.putImageData(overlay, 0, 0);
    }
  }
  if (state.registeredOverlay) {
    const registeredOffset = vectorSub(registeredCenter, center);
    const registeredX = middle + vectorDot(registeredOffset, u) / spacing;
    const registeredY = middle + vectorDot(registeredOffset, v) / spacing;
    drawRegisteredMarker(context, registeredX, registeredY);
  }
  $("perpLabel").textContent = `${Math.round(100 * fraction)}% along strut`;
  $("perpReadout").textContent =
    `Area-equivalent radius ${fmt(radius, 2)} vox · area ${fmt(area, 2)} voxels²` +
    (profileItem ? ` · tracker ${fmt(100 * profileItem.confidence, 0)}%` : "");
}

function drawAllViews() {
  if (!state.volume) return;
  drawOrthogonal("xy");
  drawOrthogonal("xz");
  drawOrthogonal("yz");
  drawPerpendicular();
}

function updateOverlayButton(id, label, enabled) {
  const button = $(id);
  button.setAttribute("aria-pressed", String(enabled));
  button.innerHTML = `<span></span> ${label}: ${enabled ? "on" : "off"}`;
}

function toggleTrackedOverlay() {
  state.trackedOverlay = !state.trackedOverlay;
  updateOverlayButton("overlayToggle", "CT tracking", state.trackedOverlay);
  drawAllViews();
}

function toggleRegisteredOverlay() {
  state.registeredOverlay = !state.registeredOverlay;
  updateOverlayButton(
    "registeredToggle",
    "Registered position",
    state.registeredOverlay
  );
  drawAllViews();
}

function closeModal() {
  $("viewerModal").hidden = true;
  document.body.style.overflow = "";
  state.volume = null;
}

Object.values(inputs).forEach((input) => input.addEventListener("change", refreshFileState));
$("loadFiles").addEventListener("click", loadFiles);
$("clearFiles").addEventListener("click", clearFiles);
$("searchInput").addEventListener("input", renderStrutList);
$("defectFilter").addEventListener("change", renderStrutList);
$("openViews").addEventListener("click", openFourViews);
$("overlayToggle").addEventListener("click", toggleTrackedOverlay);
$("registeredToggle").addEventListener("click", toggleRegisteredOverlay);
$("closeModal").addEventListener("click", closeModal);
$("viewerModal").addEventListener("click", (event) => {
  if (event.target === $("viewerModal")) closeModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("viewerModal").hidden) closeModal();
});
["xySlider", "xzSlider", "yzSlider"].forEach((id) => {
  $(id).addEventListener("input", () => {
    drawOrthogonal(id.slice(0, 2));
  });
});
$("perpSlider").addEventListener("input", drawPerpendicular);

refreshFileState();
