"use strict";

const $ = (s) => document.querySelector(s);
const TARGET_RATE = 22050;          // must match the model's training sample rate
const WINDOW_SAMPLES = TARGET_RATE; // 1-second model window
const CIRC = 2 * Math.PI * 52;

/* ─── Theme ─────────────────────────────────────────────────────────── */
const themeBtn = $("#themeToggle");
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  themeBtn.querySelector(".theme-icon").textContent = t === "dark" ? "🌙" : "☀️";
  localStorage.setItem("aids-theme", t);
}
applyTheme(
  localStorage.getItem("aids-theme") ||
  (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
);
themeBtn.addEventListener("click", () => {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
});

/* ─── Tabs ──────────────────────────────────────────────────────────── */
function activateTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll("[data-panel]").forEach((p) =>
    p.classList.toggle("hidden", p.dataset.panel !== name)
  );
  // Detection log only makes sense for live monitoring.
  $("#logWrap").classList.toggle("hidden", name !== "live");
  if (name !== "live") stopLive();
}
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => activateTab(tab.dataset.tab))
);
// Hero and navbar "launch" buttons jump into the detector on a chosen tab.
document.querySelectorAll("[data-launch]").forEach((el) =>
  el.addEventListener("click", () => activateTab(el.dataset.tab || "live"))
);

/* ─── Threshold ─────────────────────────────────────────────────────── */
const threshSlider = $("#threshold");
function syncThreshold() {
  const v = parseInt(threshSlider.value, 10);
  $("#threshVal").textContent = v;
  $("#threshLevel").textContent = v <= 30 ? "Lenient" : v >= 70 ? "Strict" : "Balanced";
}
function stepThreshold(delta) {
  const step = parseInt(threshSlider.step, 10) || 5;
  const min = parseInt(threshSlider.min, 10);
  const max = parseInt(threshSlider.max, 10);
  const next = Math.min(max, Math.max(min, parseInt(threshSlider.value, 10) + delta * step));
  threshSlider.value = next;
  syncThreshold();
}
threshSlider.addEventListener("input", syncThreshold);
$("#threshDown").addEventListener("click", () => stepThreshold(-1));
$("#threshUp").addEventListener("click", () => stepThreshold(1));
syncThreshold();
const threshold = () => parseInt(threshSlider.value, 10) / 100;

/* ─── Status / health ───────────────────────────────────────────────── */
function setStatus(cls, text) {
  const pill = $("#statusPill");
  pill.className = "pill " + cls;
  $("#statusText").textContent = text;
}
fetch("/api/health")
  .then((r) => r.json())
  .then((d) => d.models_available ? setStatus("pill-ok", "Models online") : setStatus("pill-err", "Models unavailable"))
  .catch(() => setStatus("pill-err", "Backend offline"));

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 4500);
}

/* ─── Result rendering ──────────────────────────────────────────────── */
function renderResult(res, detail) {
  $("#resultEmpty").classList.add("hidden");
  const view = $("#resultView");
  view.classList.remove("hidden");
  view.classList.toggle("threat", res.is_intrusion);

  const pct = Math.round(res.binary_prob * 100);
  $("#gaugePct").textContent = pct + "%";
  const arc = $("#gaugeArc");
  arc.style.strokeDashoffset = CIRC * (1 - res.binary_prob);
  arc.style.stroke = res.is_intrusion ? "var(--threat)" : "var(--safe)";

  $("#verdictKicker").textContent = res.is_intrusion ? "⚠ Threat detected" : "✓ All clear";
  $("#verdictMain").textContent = res.is_intrusion ? res.label : "Normal environment";
  $("#verdictDetail").textContent = detail || "";

  const bd = $("#breakdown");
  if (res.multiclass_probs) {
    bd.classList.remove("hidden");
    const order = Object.entries(res.multiclass_probs).sort((a, b) => b[1] - a[1]);
    $("#breakdownBars").innerHTML = order
      .map(([k, v]) => {
        const p = Math.round(v * 100);
        return `<div class="bar-row"><div class="bar-top"><span>${labelFor(k)}</span><span>${p}%</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:${p}%"></div></div></div>`;
      })
      .join("");
  } else {
    bd.classList.add("hidden");
  }
}

const LABELS = {
  glass_breaking: "🪟 Glass Breaking", gun_shot: "🔫 Gun Shot",
  drilling: "🔧 Drilling", jackhammer: "⚒️ Jackhammer", normal: "✅ Normal",
};
const labelFor = (k) => LABELS[k] || k;

function logDetection(res) {
  const list = $("#log");
  const empty = list.querySelector(".log-empty");
  if (empty) empty.remove();
  const li = document.createElement("li");
  li.className = "log-item " + (res.is_intrusion ? "threat" : "safe");
  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  li.innerHTML = `<span class="log-time">${time}</span>
    <span class="log-label">${res.is_intrusion ? labelFor(res.predicted_class) : "✅ Normal"}</span>
    <span class="log-prob" style="color:${res.is_intrusion ? "var(--threat)" : "var(--safe)"}">
      ${Math.round(res.binary_prob * 100)}%</span>`;
  list.prepend(li);
  while (list.children.length > 50) list.lastChild.remove();
}
$("#clearLog").addEventListener("click", () => ($("#log").innerHTML = ""));

/* ─── Audio helpers ─────────────────────────────────────────────────── */
function downsample(buf, inRate, outRate) {
  if (outRate >= inRate) return Float32Array.from(buf);
  const ratio = inRate / outRate;
  const out = new Float32Array(Math.round(buf.length / ratio));
  let pos = 0;
  for (let i = 0; i < out.length; i++) {
    const next = Math.round((i + 1) * ratio);
    let sum = 0, n = 0;
    for (let j = pos; j < next && j < buf.length; j++) { sum += buf[j]; n++; }
    out[i] = n ? sum / n : 0;
    pos = next;
  }
  return out;
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE");
  w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true); off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function analyzePcm(float32) {
  const res = await fetch(`/api/analyze-pcm?threshold=${threshold()}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: float32.buffer,
  });
  if (!res.ok) throw new Error((await res.json()).error || "Request failed");
  return res.json();
}

/* ─── Audio engine (mic capture → 16 kHz PCM + waveform) ────────────── */
class AudioEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.buffer = [];      // resampled 16 kHz samples
    this.active = false;
    this.keepAll = false;  // record mode keeps everything; live trims
  }
  async start({ keepAll = false, onLevel } = {}) {
    this.keepAll = keepAll;
    this.buffer = [];
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.proc = this.ctx.createScriptProcessor(4096, 1, 1);

    this.proc.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const ds = downsample(input, this.ctx.sampleRate, TARGET_RATE);
      for (let i = 0; i < ds.length; i++) this.buffer.push(ds[i]);
      if (!this.keepAll && this.buffer.length > TARGET_RATE * 5) {
        this.buffer.splice(0, this.buffer.length - TARGET_RATE * 5);
      }
      if (onLevel) {
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        onLevel(Math.min(1, Math.sqrt(sum / input.length) * 3));
      }
    };

    this.source.connect(this.analyser);
    this.analyser.connect(this.proc);
    this.proc.connect(this.ctx.destination);
    this.active = true;
    this._draw();
  }
  lastWindow() { return Float32Array.from(this.buffer.slice(-WINDOW_SAMPLES)); }
  allSamples() { return Float32Array.from(this.buffer); }
  _draw() {
    if (!this.active || !this.canvas) return;
    const cv = this.canvas, c = cv.getContext("2d");
    cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio;
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(data);
    c.clearRect(0, 0, cv.width, cv.height);
    c.lineWidth = 2 * devicePixelRatio;
    c.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    c.beginPath();
    const slice = cv.width / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * (cv.height / 2);
      i === 0 ? c.moveTo(0, y) : c.lineTo(i * slice, y);
    }
    c.stroke();
    this._raf = requestAnimationFrame(() => this._draw());
  }
  stop() {
    this.active = false;
    cancelAnimationFrame(this._raf);
    try { this.proc && this.proc.disconnect(); } catch (e) {}
    try { this.source && this.source.disconnect(); } catch (e) {}
    try { this.ctx && this.ctx.close(); } catch (e) {}
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
  }
}

/* ─── Live monitor ──────────────────────────────────────────────────── */
let liveEngine = null, liveTimer = null, liveBusy = false;
const liveBtn = $("#liveToggle");

liveBtn.addEventListener("click", () => (liveEngine ? stopLive() : startLive()));

async function startLive() {
  try {
    liveEngine = new AudioEngine($("#waveform"));
    await liveEngine.start({ onLevel: (l) => ($("#meterFill").style.width = (l * 100) + "%") });
    liveBtn.classList.add("recording");
    $("#liveLabel").textContent = "Stop monitoring";
    $("#liveHint").textContent = "Listening… scanning every 2 seconds.";
    setStatus("pill-live", "Monitoring");
    if (!$("#log").children.length) $("#log").innerHTML = '<li class="log-empty">Waiting for the first scan…</li>';
    liveTimer = setInterval(scanLive, 2000);
  } catch (e) {
    toast("Microphone access denied or unavailable.");
    liveEngine = null;
  }
}

async function scanLive() {
  if (!liveEngine || liveBusy || liveEngine.buffer.length < TARGET_RATE) return;
  liveBusy = true;
  try {
    const res = await analyzePcm(liveEngine.lastWindow());
    if (res.error) throw new Error(res.error);
    renderResult(res, "Live capture · last 2 seconds.");
    logDetection(res);
  } catch (e) {
    console.error(e);
  } finally {
    liveBusy = false;
  }
}

function stopLive() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (liveEngine) { liveEngine.stop(); liveEngine = null; }
  liveBtn.classList.remove("recording");
  $("#liveLabel").textContent = "Start monitoring";
  $("#liveHint").textContent = "Microphone access is requested when you start.";
  $("#meterFill").style.width = "0%";
  setStatus("pill-ok", "Models online");
}

/* ─── Record ────────────────────────────────────────────────────────── */
let recEngine = null;
const recBtn = $("#recToggle");

recBtn.addEventListener("click", async () => {
  if (recEngine) {
    const samples = recEngine.allSamples();
    recEngine.stop(); recEngine = null;
    recBtn.classList.remove("recording");
    $("#recLabel").textContent = "Start recording";
    if (samples.length < TARGET_RATE * 0.3) { toast("Recording too short, try again."); return; }
    const url = URL.createObjectURL(encodeWav(samples, TARGET_RATE));
    const prev = $("#recPreview"); prev.src = url; prev.classList.remove("hidden");
    const btn = $("#analyzeRecBtn"); btn.classList.remove("hidden");
    btn.onclick = async () => {
      btn.disabled = true; btn.textContent = "Analyzing…";
      try {
        const res = await analyzePcm(samples);
        if (res.error) throw new Error(res.error);
        renderResult(res, `Recording length: ${(samples.length / TARGET_RATE).toFixed(1)}s.`);
      } catch (e) { toast(e.message); }
      btn.disabled = false; btn.textContent = "🔍 Analyze recording";
    };
  } else {
    try {
      recEngine = new AudioEngine($("#recWaveform"));
      await recEngine.start({ keepAll: true });
      recBtn.classList.add("recording");
      $("#recLabel").textContent = "Stop recording";
      $("#recPreview").classList.add("hidden");
      $("#analyzeRecBtn").classList.add("hidden");
    } catch (e) { toast("Microphone access denied or unavailable."); recEngine = null; }
  }
});

/* ─── Upload ────────────────────────────────────────────────────────── */
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
let selectedFile = null;

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  const prev = $("#filePreview");
  prev.src = URL.createObjectURL(f); prev.classList.remove("hidden");
  $("#dropzone").querySelector(".dz-title").textContent = f.name;
  $("#analyzeFileBtn").classList.remove("hidden");
}

$("#analyzeFileBtn").addEventListener("click", async () => {
  if (!selectedFile) return;
  const btn = $("#analyzeFileBtn");
  btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("threshold", threshold());
    const res = await fetch("/api/analyze-file", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    const detail = data.n_flagged
      ? `Flagged ${data.n_flagged} of ${data.n_windows} window(s) across ${data.duration}s.`
      : `Scanned ${data.n_windows} window(s) across ${data.duration}s.`;
    renderResult(data, detail);
  } catch (e) { toast(e.message); }
  btn.disabled = false; btn.textContent = "🔍 Analyze audio";
});

/* ─── Demo samples ──────────────────────────────────────────────────── */
async function runSample(url, label) {
  setStatus("pill-live", "Analyzing sample…");
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    const name = url.split("/").pop();
    const file = new File([buf], name, { type: "audio/wav" });
    const prev = $("#filePreview");
    prev.src = URL.createObjectURL(file);
    prev.classList.remove("hidden");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("threshold", threshold());
    const res = await fetch("/api/analyze-file", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    const detail = data.n_flagged
      ? `Sample "${label}" flagged ${data.n_flagged} of ${data.n_windows} window(s).`
      : `Sample "${label}" scanned ${data.n_windows} window(s), no threat.`;
    renderResult(data, detail);
  } catch (e) {
    toast(e.message || "Could not load sample.");
  } finally {
    setStatus("pill-ok", "Models online");
  }
}
document.querySelectorAll(".sample-btn").forEach((btn) =>
  btn.addEventListener("click", () => {
    activateTab("upload");
    runSample(btn.dataset.sample, btn.textContent.trim());
  })
);

// Default tab = live, so hide log only when away.
$("#logWrap").classList.remove("hidden");
