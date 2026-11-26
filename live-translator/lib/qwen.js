// Клиент Qwen LiveTranslate (Alibaba Cloud Model Studio, realtime WebSocket).
//
// Протокол взят из рабочих открытых клиентов (ReaLingo, qwen-livetranslate-voice-clone):
//   wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=<model>
//   заголовок Authorization: Bearer <DASHSCOPE_API_KEY>
//   -> session.update, затем input_audio_buffer.append (PCM16 mono 16 кГц, base64)
//   <- текст оригинала, текст перевода (text = подтверждённая часть, stash = черновик),
//      response.audio.delta (PCM16 mono 24 кГц, base64)
import { randomUUID } from "node:crypto";
import WebSocket from "ws";

export const QWEN_HOSTS = {
  intl: "dashscope-intl.aliyuncs.com", // Сингапур — для аккаунтов alibabacloud.com
  cn: "dashscope.aliyuncs.com", // Пекин — для аккаунтов aliyun.com
};

export const ASR_MODEL = "qwen3-asr-flash-realtime";
export const INPUT_RATE = 16000;
export const OUTPUT_RATE = 24000;

export function qwenUrl({ model, region = "intl", apiHost, override }) {
  if (override) return `${override}${override.includes("?") ? "&" : "?"}model=${encodeURIComponent(model)}`;
  // API Host рабочего пространства (ws-xxxx.ap-southeast-1.maas.aliyuncs.com) — для ключей вида sk-ws-…
  const clean = (apiHost || "").trim().replace(/^[a-z]+:\/\//i, "").replace(/\/.*$/, "");
  const host = clean || QWEN_HOSTS[region] || QWEN_HOSTS.intl;
  return `wss://${host}/api-ws/v1/realtime?model=${encodeURIComponent(model)}`;
}

/**
 * Собирает session.update.
 * @param {object} o
 * @param {string} o.source  язык речи (ru)
 * @param {string} o.target  язык перевода (en | es | ur)
 * @param {boolean} o.audio  нужна ли озвучка от Qwen (false — только текст, озвучивает ElevenLabs)
 * @param {string} o.voiceMode  "once" | "always" — клон голоса на лету; "fixed" — заранее созданный клон;
 *                              "preset" — готовый голос Qwen
 * @param {string} o.voice  имя готового голоса (preset) или ID клона (fixed)
 */
export function buildSessionUpdate({ source, target, audio, voiceMode, voice, silenceMs = 700 }) {
  const session = {
    modalities: audio ? ["text", "audio"] : ["text"],
    input_audio_format: "pcm",
    sample_rate: INPUT_RATE,
    input_audio_transcription: { model: ASR_MODEL, language: source },
    translation: {
      language: target,
      // Если микрофон слышит собственный перевод (уже на целевом языке), модель его пропустит.
      same_language_skip_options: { skip_text: true, skip_audio: true },
    },
    turn_detection: { type: "server_vad", threshold: 0.2, silence_duration_ms: silenceMs },
  };
  if (audio) {
    session.output_audio_format = "pcm";
    if (voiceMode === "once" || voiceMode === "always") {
      // Клон с живого голоса: Qwen требует voice = "default".
      session.voice = "default";
      session.enable_voice_clone = true;
      session.voice_clone_options = { frequency: voiceMode };
    } else if (voiceMode === "fixed") {
      session.voice = voice;
      session.enable_voice_clone = true;
      session.voice_clone_options = { frequency: "never" };
    } else {
      session.voice = voice || "Alek";
      session.enable_voice_clone = false;
    }
  }
  return { event_id: `event_${randomUUID().replace(/-/g, "")}`, type: "session.update", session };
}

/**
 * Превращает событие Qwen в простое событие для приложения (или null).
 * Возвращает {kind:"source"|"target"|"audio"|"speech"|"link"|"ready"|"finished"|"error", ...}
 */
export function classify(ev) {
  const t = ev?.type;
  switch (t) {
    case "session.updated":
      return { kind: "ready" };
    case "session.finished":
      return { kind: "finished" };
    case "error": {
      const e = ev.error || {};
      return { kind: "error", code: String(e.code || e.type || "error"), message: String(e.message || "Ошибка Qwen") };
    }
    case "input_audio_buffer.speech_started":
      return { kind: "speech", state: "start" };
    case "input_audio_buffer.speech_stopped":
      return { kind: "speech", state: "stop" };
    case "conversation.item.created": {
      // Элемент перевода ссылается на элемент оригинала через previous_item_id.
      const prev = ev.previous_item_id;
      const id = ev.item?.id;
      if (typeof prev === "string" && prev && typeof id === "string") return { kind: "link", id, sourceId: prev };
      return null;
    }
    case "conversation.item.input_audio_transcription.text":
      return { kind: "source", id: ev.item_id, text: ev.text || "", stash: ev.stash || "", final: false };
    case "conversation.item.input_audio_transcription.completed":
      return { kind: "source", id: ev.item_id, text: ev.transcript || "", stash: "", final: true };
    case "response.text.text":
    case "response.audio_transcript.text":
      return { kind: "target", id: ev.item_id, text: ev.text || "", stash: ev.stash || "", final: false };
    case "response.text.done":
      return { kind: "target", id: ev.item_id, text: ev.text || "", stash: "", final: true };
    case "response.audio_transcript.done":
      return { kind: "target", id: ev.item_id, text: ev.transcript || "", stash: "", final: true };
    case "response.audio.delta":
      return typeof ev.delta === "string" && ev.delta ? { kind: "audio", id: ev.item_id, b64: ev.delta } : null;
    default:
      return null;
  }
}

/** Открывает сессию Qwen. Возвращает {send(pcmBuffer), finish(), close()}. */
export function openQwen({ apiKey, url, sessionUpdate, onEvent, onClose }) {
  const ws = new WebSocket(url, { headers: { Authorization: `Bearer ${apiKey}` } });
  let open = false;
  const pending = [];

  ws.on("open", () => {
    open = true;
    ws.send(JSON.stringify(sessionUpdate));
    for (const m of pending.splice(0)) ws.send(m);
  });
  ws.on("message", (data, isBinary) => {
    if (isBinary) return;
    let ev;
    try {
      ev = JSON.parse(data.toString());
    } catch {
      return;
    }
    const c = classify(ev);
    if (c) onEvent(c);
  });
  ws.on("unexpected-response", (_req, res) => {
    let body = "";
    res.on("data", (d) => (body += d));
    res.on("end", () => onEvent({ kind: "error", code: `HTTP ${res.statusCode}`, message: body.slice(0, 300) || "Qwen отклонил подключение (проверьте ключ и регион)" }));
  });
  ws.on("error", (err) => onEvent({ kind: "error", code: "network", message: err.message }));
  ws.on("close", (code, reason) => onClose?.(code, reason?.toString()));

  const sendRaw = (obj) => {
    const m = JSON.stringify(obj);
    if (open && ws.readyState === WebSocket.OPEN) ws.send(m);
    else if (!open && pending.length < 50) pending.push(m);
  };

  return {
    send(pcm) {
      if (!pcm?.length || pcm.length % 2) return;
      sendRaw({ event_id: `event_${randomUUID().replace(/-/g, "")}`, type: "input_audio_buffer.append", audio: pcm.toString("base64") });
    },
    finish() {
      sendRaw({ event_id: `event_${randomUUID().replace(/-/g, "")}`, type: "session.finish" });
    },
    close() {
      try {
        ws.close();
      } catch {
        /* уже закрыт */
      }
    },
  };
}
