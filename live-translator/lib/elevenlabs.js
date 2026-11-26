// Озвучка перевода через ElevenLabs вашим клонированным голосом.
//
// Два транспорта (оба отдают PCM16 mono 24 кГц, как и Qwen):
//  * eleven_v4_turbo — WebSocket Text-to-Dialogue (быстрый старт, хорошо держит клон);
//    одна сессия заранее открыта и «прогрета», чтобы не тратить время на рукопожатие;
//  * любая другая модель (eleven_flash_v2_5, eleven_multilingual_v2, …) — HTTP-стриминг.
//
// Каждое предложение — отдельная задача. Запросы идут параллельно, а звук уходит
// на телефон строго по порядку.
import WebSocket from "ws";

const KEEP_ALIVE_MS = 10_000;

export class ElevenLabsSpeaker {
  /**
   * @param {object} o
   * @param {string} o.apiKey
   * @param {string} o.voiceId
   * @param {string} o.model
   * @param {string} o.language  код языка перевода (en, es, ur)
   * @param {(pcm: Buffer) => void} o.onAudio
   * @param {(err: Error) => void} o.onError
   * @param {string} [o.httpBase]  для тестов
   * @param {string} [o.wsBase]    для тестов
   */
  constructor(o) {
    this.o = o;
    this.httpBase = o.httpBase || "https://api.elevenlabs.io";
    this.wsBase = o.wsBase || "wss://api.elevenlabs.io";
    this.jobs = [];
    this.closed = false;
    this.previous = null;
    this.useDialogueWs = o.model === "eleven_v4_turbo";
    this.warm = null;
    if (this.useDialogueWs) this.#prepareWarm();
  }

  speak(text) {
    if (this.closed || !text.trim()) return;
    const job = { chunks: [], done: false, carry: null, startedAt: Date.now(), firstAudioAt: 0 };
    this.jobs.push(job);
    const run = this.useDialogueWs ? this.#viaDialogueWs(text, job) : this.#viaHttp(text, job);
    run
      .catch((err) => {
        if (!this.closed) this.o.onError(err);
      })
      .finally(() => {
        job.done = true;
        this.#pump();
      });
  }

  close() {
    this.closed = true;
    this.jobs = [];
    this.warm?.then((ws) => ws?.close()).catch(() => {});
    clearInterval(this.keepAlive);
  }

  // ---- выдача звука по порядку -------------------------------------------------
  #push(job, bytes) {
    let buf = job.carry ? Buffer.concat([job.carry, bytes]) : bytes;
    job.carry = null;
    if (buf.length % 2) {
      job.carry = buf.subarray(buf.length - 1);
      buf = buf.subarray(0, buf.length - 1);
    }
    if (!buf.length) return;
    if (!job.firstAudioAt) job.firstAudioAt = Date.now();
    job.chunks.push(buf);
    this.#pump();
  }

  #pump() {
    while (this.jobs.length) {
      const head = this.jobs[0];
      for (const c of head.chunks.splice(0)) this.o.onAudio(c);
      if (!head.done) return;
      this.jobs.shift();
    }
  }

  // ---- HTTP-стриминг ------------------------------------------------------------
  async #viaHttp(text, job) {
    const body = { text, model_id: this.o.model };
    // language_code понимают модели v2.5; остальным язык не передаём, они определяют его по тексту.
    if (/_v2_5$/.test(this.o.model) && this.o.language) body.language_code = this.o.language;
    if (this.previous) body.previous_text = this.previous;
    this.previous = text;
    const url = `${this.httpBase}/v1/text-to-speech/${encodeURIComponent(this.o.voiceId)}/stream?output_format=pcm_24000`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "xi-api-key": this.o.apiKey, "content-type": "application/json", accept: "audio/pcm" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      throw new Error(`ElevenLabs ${res.status}: ${explain(t)}`);
    }
    for await (const chunk of res.body) {
      if (this.closed) return;
      this.#push(job, Buffer.from(chunk));
    }
  }

  // ---- WebSocket Text-to-Dialogue (eleven_v4_turbo) ----------------------------
  #openDialogue() {
    const url = `${this.wsBase}/v1/text-to-dialogue/stream-input?model_id=${encodeURIComponent(this.o.model)}&output_format=pcm_24000`;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.once("open", () => {
        // Первое сообщение авторизует и регистрирует единственный голос этой сессии.
        ws.send(JSON.stringify({ voices: [this.o.voiceId], xi_api_key: this.o.apiKey }));
        resolve(ws);
      });
      ws.once("unexpected-response", (_req, res) => {
        let b = "";
        res.on("data", (d) => (b += d));
        res.on("end", () => reject(new Error(`ElevenLabs отклонил WebSocket: HTTP ${res.statusCode} ${explain(b)}`)));
      });
      ws.once("error", reject);
    });
  }

  #prepareWarm() {
    if (this.closed) return;
    this.warm = this.#openDialogue().catch((err) => {
      this.warmError = err;
      return null;
    });
    clearInterval(this.keepAlive);
    this.keepAlive = setInterval(() => {
      this.warm?.then((ws) => {
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ keep_alive: true }));
      });
    }, KEEP_ALIVE_MS);
  }

  async #takeSession() {
    // Забираем прогретую сессию синхронно, чтобы два предложения подряд не получили одну и ту же.
    const taken = this.warm;
    this.#prepareWarm();
    let ws = await taken;
    if (!ws || ws.readyState !== WebSocket.OPEN) ws = await this.#openDialogue();
    return ws;
  }

  async #viaDialogueWs(text, job) {
    const ws = await this.#takeSession();
    await new Promise((resolve, reject) => {
      ws.on("message", (data) => {
        let v;
        try {
          v = JSON.parse(data.toString());
        } catch {
          return;
        }
        if (v.error) {
          reject(new Error(`ElevenLabs: ${v.message || JSON.stringify(v.error)}`));
          ws.close();
          return;
        }
        if (typeof v.audio === "string" && v.audio) this.#push(job, Buffer.from(v.audio, "base64"));
        if (v.is_final === true || v.isFinal === true) {
          resolve();
          ws.close();
        }
      });
      ws.on("close", () => resolve());
      ws.on("error", reject);
      ws.send(JSON.stringify({ inputs: [{ text, voice_id: this.o.voiceId, new_turn: false }] }));
      ws.send(JSON.stringify({ close_socket: true }));
    });
  }
}

function explain(body) {
  try {
    const v = JSON.parse(body);
    return v?.detail?.message || v?.detail?.status || JSON.stringify(v.detail ?? v).slice(0, 300);
  } catch {
    return String(body).slice(0, 300);
  }
}
