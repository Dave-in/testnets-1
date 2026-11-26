// Сервер синхронного перевода.
//
// Телефон (веб-приложение) ⇄ этот сервер ⇄ Qwen LiveTranslate (+ ElevenLabs).
// Ключи API хранятся только здесь, в переменных окружения, и никогда не попадают в браузер.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { timingSafeEqual, createHash } from "node:crypto";
import { WebSocketServer } from "ws";
import { buildSessionUpdate, openQwen, qwenUrl } from "./lib/qwen.js";
import { SentenceFeeder } from "./lib/sentences.js";
import { ElevenLabsSpeaker } from "./lib/elevenlabs.js";

const env = process.env;
const CFG = {
  port: Number(env.PORT || 8080),
  password: env.APP_PASSWORD || "",
  qwenKey: env.DASHSCOPE_API_KEY || "",
  qwenRegion: env.QWEN_REGION || "intl",
  qwenApiHost: env.QWEN_API_HOST || "",
  qwenModel: env.QWEN_MODEL || "qwen3.5-livetranslate-flash-realtime",
  qwenUrlOverride: env.QWEN_WS_URL || "",
  qwenClonedVoice: env.QWEN_VOICE_ID || "",
  elevenKey: env.ELEVENLABS_API_KEY || "",
  elevenVoice: env.ELEVENLABS_VOICE_ID || "",
  elevenModel: env.ELEVENLABS_MODEL || "eleven_v4_turbo",
  elevenHttpBase: env.ELEVENLABS_HTTP_BASE || "",
  elevenWsBase: env.ELEVENLABS_WS_BASE || "",
  maxMinutes: Number(env.MAX_SESSION_MINUTES || 60),
};

const TARGETS = { en: "English", es: "Español", ur: "اردو" };
const PRESET_VOICES = ["Alek", "Harvey", "Andre", "Ethan", "Aiden", "Bodega", "Tina", "Serena", "Jennifer", "Sonrisa", "Marina", "Katerina"];

const PUBLIC = join(fileURLToPath(new URL(".", import.meta.url)), "public");
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".webmanifest": "application/manifest+json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

function passwordOk(given) {
  if (!CFG.password) return true;
  const a = createHash("sha256").update(String(given || "")).digest();
  const b = createHash("sha256").update(CFG.password).digest();
  return timingSafeEqual(a, b);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/config") {
    res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
    res.end(
      JSON.stringify({
        passwordRequired: Boolean(CFG.password),
        qwen: Boolean(CFG.qwenKey),
        elevenlabs: Boolean(CFG.elevenKey && CFG.elevenVoice),
        elevenModel: CFG.elevenModel,
        qwenClonedVoice: Boolean(CFG.qwenClonedVoice),
        targets: TARGETS,
        presetVoices: PRESET_VOICES,
      })
    );
    return;
  }
  if (url.pathname === "/healthz") {
    res.writeHead(200).end("ok");
    return;
  }
  let path = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, "");
  if (!path || path.endsWith("/")) path += "index.html";
  if (path.includes("..")) {
    res.writeHead(400).end();
    return;
  }
  try {
    const body = await readFile(join(PUBLIC, path));
    res.writeHead(200, {
      "content-type": MIME[extname(path)] || "application/octet-stream",
      "cache-control": path === "sw.js" || path === "index.html" ? "no-cache" : "public, max-age=300",
    });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

const wss = new WebSocketServer({ server, path: "/ws", maxPayload: 1 << 20 });

wss.on("connection", (client) => {
  let qwen = null;
  let speaker = null;
  let feeder = null;
  let timer = null;
  let turnStart = 0;
  let turnAudioReported = false;
  const reportedItems = new Set();

  const send = (obj) => client.readyState === 1 && client.send(JSON.stringify(obj));
  const sendAudio = (pcm) => client.readyState === 1 && client.send(pcm, { binary: true });

  const stop = (reason) => {
    clearTimeout(timer);
    timer = null;
    if (qwen) {
      qwen.finish();
      const q = qwen;
      setTimeout(() => q.close(), 1500);
    }
    speaker?.close();
    qwen = speaker = feeder = null;
    if (reason) send({ type: "stopped", reason });
  };

  const latency = (stage) => {
    if (!turnStart || turnAudioReported) return;
    turnAudioReported = true;
    send({ type: "latency", stage, ms: Date.now() - turnStart });
  };

  client.on("message", (data, isBinary) => {
    if (isBinary) {
      qwen?.send(Buffer.from(data));
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "stop") return stop("user");
    if (msg.type !== "start") return;

    if (!passwordOk(msg.password)) return send({ type: "error", code: "auth", message: "Неверный пароль приложения" });
    if (!CFG.qwenKey) return send({ type: "error", code: "config", message: "На сервере не задан DASHSCOPE_API_KEY" });
    const target = TARGETS[msg.target] ? msg.target : "en";
    const mode = msg.mode === "elevenlabs" ? "elevenlabs" : "qwen";
    if (mode === "elevenlabs" && !(CFG.elevenKey && CFG.elevenVoice))
      return send({ type: "error", code: "config", message: "На сервере не заданы ELEVENLABS_API_KEY и ELEVENLABS_VOICE_ID" });

    stop();
    let voiceMode = ["once", "always", "fixed", "preset"].includes(msg.voiceMode) ? msg.voiceMode : "once";
    if (voiceMode === "fixed" && !CFG.qwenClonedVoice) voiceMode = "once";
    const voice = voiceMode === "fixed" ? CFG.qwenClonedVoice : PRESET_VOICES.includes(msg.voice) ? msg.voice : "Alek";

    if (mode === "elevenlabs") {
      speaker = new ElevenLabsSpeaker({
        apiKey: CFG.elevenKey,
        voiceId: CFG.elevenVoice,
        model: CFG.elevenModel,
        language: target,
        httpBase: CFG.elevenHttpBase,
        wsBase: CFG.elevenWsBase,
        onAudio: (pcm) => {
          latency("elevenlabs");
          sendAudio(pcm);
        },
        onError: (err) => send({ type: "error", code: "elevenlabs", message: err.message }),
      });
      const sp = speaker;
      feeder = new SentenceFeeder((text) => sp.speak(text));
    }

    const update = buildSessionUpdate({ target, audio: mode === "qwen", voiceMode, voice });
    const fd = feeder;
    let mine = null;
    mine = qwen = openQwen({
      apiKey: CFG.qwenKey,
      url: qwenUrl({ model: CFG.qwenModel, region: CFG.qwenRegion, apiHost: CFG.qwenApiHost, override: CFG.qwenUrlOverride }),
      sessionUpdate: update,
      onEvent: (ev) => {
        if (qwen !== mine) return; // событие от уже остановленной сессии
        switch (ev.kind) {
          case "ready":
            return send({ type: "ready", mode, target, voiceMode });
          case "speech":
            if (ev.state === "start") {
              turnStart = Date.now();
              turnAudioReported = false;
            }
            return send({ type: "speech", state: ev.state });
          case "link":
            return send({ type: "link", id: ev.id, sourceId: ev.sourceId });
          case "source":
            return send({ type: "source", id: ev.id, text: ev.text, stash: ev.stash, final: ev.final });
          case "target":
            fd?.update(ev.id, ev.text, ev.final);
            return send({ type: "target", id: ev.id, text: ev.text, stash: ev.stash, final: ev.final });
          case "audio":
            if (mode !== "qwen") return;
            if (ev.id && !reportedItems.has(ev.id)) {
              reportedItems.add(ev.id);
              latency("qwen");
            }
            return sendAudio(Buffer.from(ev.b64, "base64"));
          case "error":
            return send({ type: "error", code: ev.code, message: ev.message });
          case "finished":
            return;
        }
      },
      onClose: (code) => {
        if (qwen && qwen === mine) {
          clearTimeout(timer);
          send({ type: "stopped", reason: `qwen-closed:${code}` });
          speaker?.close();
          qwen = speaker = feeder = null;
        }
      },
    });
    timer = setTimeout(() => stop("time-limit"), CFG.maxMinutes * 60_000);
  });

  client.on("close", () => stop());
});

server.listen(CFG.port, () => {
  console.log(`live-translator: http://localhost:${CFG.port}`);
  if (!CFG.password) console.warn("ВНИМАНИЕ: APP_PASSWORD не задан — любой, кто знает адрес, сможет тратить ваши ключи.");
  if (!CFG.qwenKey) console.warn("ВНИМАНИЕ: DASHSCOPE_API_KEY не задан — перевод работать не будет.");
  if (!CFG.elevenKey || !CFG.elevenVoice) console.log("ElevenLabs не настроен — доступен только режим «голос Qwen».");
});
