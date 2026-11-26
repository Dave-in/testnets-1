// Веб-приложение для телефона: микрофон → сервер → перевод (текст + голос) → динамик.
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) {
    try {
      const v = localStorage.getItem("lt." + k);
      return v === null ? d : JSON.parse(v);
    } catch {
      return d;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem("lt." + k, JSON.stringify(v));
    } catch {
      /* приватный режим */
    }
  },
};

const VOICE_MODES = [
  ["once", "Клон моего голоса (с первой фразы)"],
  ["always", "Клон моего голоса (подстраивается всё время)"],
];

let cfg = null;
let state = { target: store.get("target", "en"), mode: store.get("mode", "qwen"), voice: store.get("voice", "once") };

// ---- настройка интерфейса ----------------------------------------------------------
async function init() {
  try {
    cfg = await (await fetch("api/config", { cache: "no-store" })).json();
  } catch {
    toast("Сервер недоступен");
    return;
  }
  const seg = $("targets");
  for (const [code, name] of Object.entries(cfg.targets)) {
    const b = document.createElement("button");
    b.type = "button";
    b.role = "radio";
    b.textContent = name;
    b.dataset.code = code;
    b.onclick = () => {
      state.target = code;
      store.set("target", code);
      renderTargets();
      restartIfLive();
    };
    seg.append(b);
  }
  if (!cfg.targets[state.target]) state.target = "en";
  renderTargets();

  if (!cfg.elevenlabs) {
    const opt = $("elevenOption").querySelector("input");
    opt.disabled = true;
    $("elevenHint").textContent = "Не настроено на сервере: нужны ELEVENLABS_API_KEY и ELEVENLABS_VOICE_ID";
    if (state.mode === "elevenlabs") state.mode = "qwen";
  }
  for (const r of document.querySelectorAll('input[name="mode"]')) {
    r.checked = r.value === state.mode;
    r.onchange = () => {
      state.mode = r.value;
      store.set("mode", r.value);
      renderMode();
      restartIfLive();
    };
  }

  const sel = $("voice");
  const add = (value, text, parent = sel) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    parent.append(o);
  };
  for (const [v, t] of VOICE_MODES) add(v, t);
  if (cfg.qwenClonedVoice) add("fixed", "Мой заранее созданный клон");
  const g = document.createElement("optgroup");
  g.label = "Готовые голоса";
  for (const v of cfg.presetVoices) add("preset:" + v, v, g);
  sel.append(g);
  sel.value = [...sel.options].some((o) => o.value === state.voice) ? state.voice : "once";
  sel.onchange = () => {
    state.voice = sel.value;
    store.set("voice", sel.value);
    restartIfLive();
  };
  renderMode();

  $("pwRow").hidden = !cfg.passwordRequired;
  $("password").value = store.get("password", "");
  $("password").onchange = (e) => store.set("password", e.target.value);
  $("duplexMute").checked = store.get("duplexMute", false);
  $("duplexMute").onchange = (e) => store.set("duplexMute", e.target.checked);
  $("clearLog").onclick = () => {
    $("log").textContent = "";
    turns.clear();
  };
  $("micBtn").onclick = () => (live ? stop() : start());
  if (!cfg.qwen) toast("На сервере не задан ключ Qwen (DASHSCOPE_API_KEY)");
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
}

function renderTargets() {
  for (const b of $("targets").children) b.setAttribute("aria-checked", String(b.dataset.code === state.target));
}
function renderMode() {
  $("qwenVoiceRow").hidden = state.mode !== "qwen";
}

// ---- сессия ------------------------------------------------------------------------
let live = false;
let ws = null;
let ctx = null;
let micStream = null;
let worklet = null;
let wakeLock = null;
let playHead = 0;
let ready = false;
let workletLoaded = false;

async function start() {
  if (live) return;
  if (cfg?.passwordRequired && !$("password").value) {
    document.querySelector("details.settings").open = true;
    $("password").focus();
    return toast("Введите пароль приложения в настройках");
  }
  setStatus("connecting", "Подключение…");
  $("micBtn").disabled = true;
  try {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    await ctx.resume();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    if (!workletLoaded) {
      await ctx.audioWorklet.addModule("capture-worklet.js");
      workletLoaded = true;
    }
    const src = ctx.createMediaStreamSource(micStream);
    worklet = new AudioWorkletNode(ctx, "capture-processor");
    const sink = ctx.createGain();
    sink.gain.value = 0;
    src.connect(worklet).connect(sink).connect(ctx.destination);
    worklet.port.onmessage = (e) => {
      $("ring").style.setProperty("--lvl", Math.min(1, e.data.peak * 3).toFixed(2));
      if (ready && ws?.readyState === 1) ws.send(e.data.pcm);
      updateDuplexMute();
    };
  } catch (err) {
    $("micBtn").disabled = false;
    cleanupAudio();
    setStatus("error", "Нет микрофона");
    return toast("Нет доступа к микрофону: " + (err.message || err));
  }

  live = true;
  ready = false;
  $("micBtn").disabled = false;
  $("micBtn").setAttribute("aria-pressed", "true");
  $("micBtn").setAttribute("aria-label", "Остановить перевод");
  $("hint").textContent = "Подключение к переводчику…";
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch {
    /* не поддерживается */
  }
  connect();
}

function connect() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + location.pathname.replace(/[^/]*$/, "") + "ws";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    const [voiceMode, voice] = state.voice.startsWith("preset:") ? ["preset", state.voice.slice(7)] : [state.voice, ""];
    ws.send(JSON.stringify({ type: "start", password: $("password").value, target: state.target, mode: state.mode, voiceMode, voice }));
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : playPcm(e.data));
  ws.onclose = () => {
    if (live) {
      stop();
      toast("Соединение с сервером прервалось");
    }
  };
}

function stop() {
  live = false;
  ready = false;
  if (ws) {
    try {
      ws.send(JSON.stringify({ type: "stop" }));
    } catch {
      /* закрыт */
    }
    const w = ws;
    ws = null;
    w.onclose = null;
    setTimeout(() => w.close(), 1500);
  }
  cleanupAudio();
  wakeLock?.release?.().catch(() => {});
  wakeLock = null;
  $("micBtn").setAttribute("aria-pressed", "false");
  $("micBtn").setAttribute("aria-label", "Начать перевод");
  $("ring").style.setProperty("--lvl", 0);
  $("hint").textContent = "Нажмите и говорите по-русски";
  setStatus("idle", "Готов");
}

function cleanupAudio() {
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
  try {
    worklet?.disconnect();
  } catch {
    /* уже отключён */
  }
  worklet = null;
}

function restartIfLive() {
  if (!live) return;
  stop();
  setTimeout(start, 300);
}

// ---- события сервера ---------------------------------------------------------------
const turns = new Map(); // sourceId -> {el, src, dst}
const targetToSource = new Map();
const pendingTarget = new Map();

function onEvent(m) {
  switch (m.type) {
    case "ready":
      ready = true;
      $("toast").hidden = true;
      setStatus("live", "Перевожу");
      $("hint").textContent =
        m.mode === "qwen" && (m.voiceMode === "once" || m.voiceMode === "always")
          ? "Говорите. Первая фраза уйдёт на клонирование голоса"
          : "Говорите по-русски";
      break;
    case "speech":
      $("hint").textContent = m.state === "start" ? "Слушаю…" : "Перевожу…";
      break;
    case "latency":
      $("latency").hidden = false;
      $("latency").textContent = `Задержка ${(m.ms / 1000).toFixed(1)} с · ${m.stage === "qwen" ? "Qwen" : "ElevenLabs"}`;
      break;
    case "link": {
      targetToSource.set(m.id, m.sourceId);
      const p = pendingTarget.get(m.id);
      if (p) {
        pendingTarget.delete(m.id);
        renderTarget(m.sourceId, p);
      }
      break;
    }
    case "source": {
      const t = turn(m.id);
      t.src.textContent = m.text;
      setStash(t.src, m.stash);
      break;
    }
    case "target": {
      const sid = targetToSource.get(m.id);
      if (sid) renderTarget(sid, m);
      else {
        pendingTarget.set(m.id, m);
        renderTarget("t:" + m.id, m);
      }
      break;
    }
    case "error":
      toast(m.message || "Ошибка");
      if (m.code === "auth" || m.code === "config" || /^HTTP 40/.test(m.code)) stop();
      if (/^HTTP 40/.test(m.code)) setStatus("error", "Ошибка ключа");
      break;
    case "stopped":
      if (live && m.reason !== "user") {
        stop();
        toast(m.reason === "time-limit" ? "Сессия остановлена: превышен лимит времени" : "Переводчик закрыл сессию");
      }
      break;
  }
}

function turn(id) {
  let t = turns.get(id);
  if (!t) {
    const el = document.createElement("article");
    el.className = "turn";
    const src = document.createElement("div");
    src.className = "src";
    const dst = document.createElement("div");
    dst.className = "dst";
    if (state.target === "ur") dst.dir = "rtl";
    el.append(src, dst);
    $("log").prepend(el);
    t = { el, src, dst };
    turns.set(id, t);
    while ($("log").children.length > 60) $("log").lastElementChild.remove();
  }
  return t;
}

function renderTarget(sourceId, m) {
  // Если перевод временно показывался отдельно (до связки), переносим его.
  const tmp = turns.get("t:" + m.id);
  if (tmp && sourceId !== "t:" + m.id) {
    tmp.el.remove();
    turns.delete("t:" + m.id);
  }
  const t = turn(sourceId);
  t.dst.textContent = m.text;
  setStash(t.dst, m.stash);
}

function setStash(el, stash) {
  if (!stash) return;
  const s = document.createElement("span");
  s.className = "stash";
  s.textContent = stash;
  el.append(s);
}

// ---- воспроизведение ---------------------------------------------------------------
function playPcm(buf) {
  if (!ctx || !live) return;
  const pcm = new Int16Array(buf);
  if (!pcm.length) return;
  const audio = ctx.createBuffer(1, pcm.length, 24000);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const node = ctx.createBufferSource();
  node.buffer = audio;
  node.connect(ctx.destination);
  const now = ctx.currentTime;
  if (playHead < now + 0.03) playHead = now + 0.08; // небольшой запас от рывков сети
  node.start(playHead);
  playHead += audio.duration;
  updateDuplexMute();
}

function updateDuplexMute() {
  if (!worklet || !ctx) return;
  const muted = $("duplexMute").checked && playHead > ctx.currentTime + 0.05;
  if (muted !== worklet._muted) {
    worklet._muted = muted;
    worklet.port.postMessage({ muted });
  }
}

// ---- мелочи ------------------------------------------------------------------------
function setStatus(s, text) {
  $("status").dataset.state = s;
  $("status").textContent = text;
}
let toastTimer;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 6000);
}
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && live && (!wakeLock || wakeLock.released)) {
    try {
      wakeLock = await navigator.wakeLock?.request("screen");
    } catch {
      /* ничего */
    }
  }
});

init();
