const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");

const btnCamera = document.getElementById("btnCamera");
const chkOverlay = document.getElementById("chkOverlay");
const txtWord = document.getElementById("txtWord");
const txtConf = document.getElementById("txtConf");
const barConf = document.getElementById("barConf");
const txtHint = document.getElementById("txtHint");
const serverBadge = document.getElementById("serverBadge");

const selVoice = document.getElementById("selVoice");
const btnSpeak = document.getElementById("btnSpeak");
const chkAutoSpeak = document.getElementById("chkAutoSpeak");

const rngThreshold = document.getElementById("rngThreshold");
const txtThreshold = document.getElementById("txtThreshold");

const historyEl = document.getElementById("history");
const btnClear = document.getElementById("btnClear");
const btnExport = document.getElementById("btnExport");

let camera = null;
let hands = null;
let socket = null;

let lastStableWord = null;
let history = [];

const state = {
  running: false,
  lastSendTs: 0,
  sendEveryMs: 66, // ~15 fps
  confidenceThreshold: Number(rngThreshold.value),
};

function setServerBadge(ok, text) {
  serverBadge.textContent = text;
  serverBadge.className =
    "text-xs rounded-full px-3 py-1 " +
    (ok ? "bg-emerald-500/15 text-emerald-200 border border-emerald-800" : "bg-rose-500/15 text-rose-200 border border-rose-800");
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function resizeCanvasToVideo() {
  const w = videoEl.videoWidth || 1280;
  const h = videoEl.videoHeight || 720;
  if (canvasEl.width !== w) canvasEl.width = w;
  if (canvasEl.height !== h) canvasEl.height = h;
}

function drawLandmarks(results) {
  resizeCanvasToVideo();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (!chkOverlay.checked) return;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    for (const landmarks of results.multiHandLandmarks) {
      // The video is mirrored via CSS; mirror the drawing so it aligns.
      ctx.save();
      ctx.translate(canvasEl.width, 0);
      ctx.scale(-1, 1);
      window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, { color: "#34d399", lineWidth: 3 });
      window.drawLandmarks(ctx, landmarks, { color: "#a7f3d0", lineWidth: 1, radius: 2 });
      ctx.restore();
    }
  }
}

function distance3(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function extractNormalized63(landmarks) {
  // Mirror of scripts/signtospeech.py normalization:
  // wrist(0) as origin; scale by distance to landmark 9.
  const wrist = landmarks[0];
  const ref = landmarks[9];
  let scale = distance3(wrist, ref);
  if (!scale || scale === 0) scale = 1;

  const out = [];
  for (const lm of landmarks) {
    out.push((lm.x - wrist.x) / scale);
    out.push((lm.y - wrist.y) / scale);
    out.push(((lm.z || 0) - (wrist.z || 0)) / scale);
  }
  return out;
}

function setConfidenceUI(conf) {
  const pct = Math.round(clamp01(conf) * 100);
  txtConf.textContent = `${pct}%`;
  barConf.style.width = `${pct}%`;
  barConf.className = "h-2 " + (pct >= 85 ? "bg-emerald-500" : pct >= 70 ? "bg-amber-500" : "bg-rose-500");
}

function pushHistory(word, conf) {
  history.unshift({ word, conf, ts: new Date().toISOString() });
  history = history.slice(0, 10);
  renderHistory();
}

function renderHistory() {
  historyEl.innerHTML = "";
  for (const item of history) {
    const li = document.createElement("li");
    li.className = "flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2";
    const left = document.createElement("div");
    left.innerHTML = `<div class="text-sm font-semibold">${item.word}</div><div class="text-[11px] text-zinc-500">${new Date(item.ts).toLocaleTimeString()}</div>`;
    const right = document.createElement("div");
    right.className = "flex items-center gap-2";
    const badge = document.createElement("div");
    badge.className = "text-[11px] px-2 py-1 rounded-full bg-zinc-800 text-zinc-200 tabular-nums";
    badge.textContent = `${Math.round(clamp01(item.conf) * 100)}%`;
    const btn = document.createElement("button");
    btn.className = "text-[11px] rounded-md px-2 py-1 bg-indigo-600 hover:bg-indigo-500";
    btn.textContent = "Replay";
    btn.onclick = () => speak(item.word);
    right.appendChild(badge);
    right.appendChild(btn);
    li.appendChild(left);
    li.appendChild(right);
    historyEl.appendChild(li);
  }
}

function speak(text) {
  if (!text || text === "..." || text === "…") return;
  if (!("speechSynthesis" in window)) return;

  const u = new SpeechSynthesisUtterance(text);
  const voiceName = selVoice.value;
  const voices = window.speechSynthesis.getVoices();
  const v = voices.find((x) => x.name === voiceName);
  if (v) u.voice = v;
  u.rate = 1.0;
  u.pitch = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

function loadVoices() {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  selVoice.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.name;
    opt.textContent = `${v.name} (${v.lang})`;
    selVoice.appendChild(opt);
  }
  if (voices.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No voices available";
    selVoice.appendChild(opt);
  }
}

function connectSocket() {
  socket = window.io();
  setServerBadge(false, "Server: connecting…");

  socket.on("connect", () => setServerBadge(true, "Server: connected"));
  socket.on("disconnect", () => setServerBadge(false, "Server: disconnected"));

  socket.on("server_status", (msg) => {
    if (msg?.ok) {
      setServerBadge(true, "Server: model ready");
      txtHint.textContent = "Move your hand in frame to start recognition.";
    } else {
      setServerBadge(false, "Server: model not loaded");
      txtHint.textContent = msg?.error ? `Model error: ${msg.error}` : "Model not loaded. Put files in server/models.";
    }
    if (typeof msg?.confidence_threshold === "number") {
      state.confidenceThreshold = msg.confidence_threshold;
      rngThreshold.value = String(msg.confidence_threshold);
      txtThreshold.textContent = Number(msg.confidence_threshold).toFixed(2);
    }
  });

  socket.on("prediction", (msg) => {
    if (!msg?.ok) {
      txtHint.textContent = msg?.error || "Prediction error.";
      return;
    }

    const conf = Number(msg.confidence || 0);
    setConfidenceUI(conf);

    const word = msg.word || "...";
    txtWord.textContent = word === "..." ? "…" : String(word).toUpperCase();

    if (msg.ready === false) {
      txtHint.textContent = "Collecting sequence…";
      return;
    }

    if (msg.is_stable) {
      txtHint.textContent = "Stable detection.";
      if (word !== "..." && word !== lastStableWord) {
        lastStableWord = word;
        pushHistory(String(word).toUpperCase(), conf);
        if (chkAutoSpeak.checked) speak(String(word));
      }
    } else {
      txtHint.textContent = "Tracking…";
    }
  });
}

async function startCamera() {
  if (state.running) return;
  state.running = true;
  btnCamera.textContent = "Stop camera";
  btnCamera.className = "text-xs rounded-md px-3 py-2 bg-rose-600 hover:bg-rose-500";

  if (!hands) {
    hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.75,
    });
    hands.onResults((results) => {
      drawLandmarks(results);

      if (!socket || socket.disconnected) return;
      if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

      const now = performance.now();
      if (now - state.lastSendTs < state.sendEveryMs) return;
      state.lastSendTs = now;

      const landmarks = results.multiHandLandmarks[0];
      const landmarks63 = extractNormalized63(landmarks);
      socket.emit("landmarks", { ts: Date.now(), landmarks_63: landmarks63 });
    });
  }

  camera = new window.Camera(videoEl, {
    onFrame: async () => {
      if (!state.running) return;
      await hands.send({ image: videoEl });
    },
    width: 1280,
    height: 720,
  });

  await camera.start();
  txtHint.textContent = "Camera started. Looking for hands…";
}

async function stopCamera() {
  state.running = false;
  btnCamera.textContent = "Start camera";
  btnCamera.className = "text-xs rounded-md px-3 py-2 bg-indigo-600 hover:bg-indigo-500";
  try {
    if (camera) await camera.stop();
  } catch {
    // ignore
  }
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  txtHint.textContent = "Camera stopped.";
}

btnCamera.addEventListener("click", async () => {
  if (!state.running) await startCamera();
  else await stopCamera();
});

btnSpeak.addEventListener("click", () => speak(txtWord.textContent));

rngThreshold.addEventListener("input", () => {
  const v = Number(rngThreshold.value);
  state.confidenceThreshold = v;
  txtThreshold.textContent = v.toFixed(2);
  socket?.emit("set_settings", { confidence_threshold: v });
});

btnClear.addEventListener("click", () => {
  history = [];
  renderHistory();
});

btnExport.addEventListener("click", () => {
  const lines = history
    .slice()
    .reverse()
    .map((x) => `${new Date(x.ts).toLocaleString()}  ${x.word}  (${Math.round(clamp01(x.conf) * 100)}%)`);
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sign-session.txt";
  a.click();
  URL.revokeObjectURL(url);
});

if ("speechSynthesis" in window) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}

txtThreshold.textContent = Number(rngThreshold.value).toFixed(2);
connectSocket();

