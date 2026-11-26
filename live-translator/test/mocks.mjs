// Поддельные Qwen LiveTranslate и ElevenLabs для проверки без настоящих ключей.
import { createServer } from "node:http";
import { WebSocketServer } from "ws";

export function tone(ms, rate = 24000, freq = 440) {
  const n = Math.round((rate * ms) / 1000);
  const b = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(Math.sin((2 * Math.PI * freq * i) / rate) * 8000), i * 2);
  return b;
}

/** Qwen: после ~1 с входного звука выдаёт одну реплику с переводом (и звуком, если просили). */
export function startMockQwen(port, log = {}) {
  log.sessions = [];
  log.appends = 0;
  log.auth = [];
  const wss = new WebSocketServer({ port });
  wss.on("connection", (ws, req) => {
    log.auth.push(req.headers.authorization);
    let session = null;
    let appends = 0;
    let turn = 0;
    const send = (o) => ws.send(JSON.stringify(o));
    ws.on("message", async (raw) => {
      const ev = JSON.parse(raw.toString());
      if (ev.type === "session.update") {
        session = ev.session;
        log.sessions.push(session);
        send({ type: "session.updated", session: { ...session, model: "mock" } });
        return;
      }
      if (ev.type === "session.finish") {
        send({ type: "session.finished" });
        return;
      }
      if (ev.type !== "input_audio_buffer.append" || !session) return;
      appends++;
      log.appends++;
      if (appends % 10 !== 0 || turn >= 2) return;
      turn++;
      const src = `src${turn}`;
      const tr = `tr${turn}`;
      send({ type: "input_audio_buffer.speech_started", item_id: src });
      send({ type: "conversation.item.input_audio_transcription.text", item_id: src, text: "Привет, друг.", stash: " Как дела" });
      send({ type: "conversation.item.created", previous_item_id: src, item: { id: tr } });
      const audio = session.modalities.includes("audio");
      const tt = audio ? "response.audio_transcript.text" : "response.text.text";
      send({ type: tt, item_id: tr, text: "Hello there, my friend.", stash: " How" });
      if (audio) for (let i = 0; i < 3; i++) send({ type: "response.audio.delta", item_id: tr, delta: tone(200).toString("base64") });
      await new Promise((r) => setTimeout(r, 100));
      send({ type: "conversation.item.input_audio_transcription.completed", item_id: src, transcript: "Привет, друг. Как дела?" });
      send({ type: "input_audio_buffer.speech_stopped", item_id: src });
      send({ type: tt, item_id: tr, text: "Hello there, my friend. How are you", stash: "" });
      if (audio) send({ type: "response.audio.delta", item_id: tr, delta: tone(200).toString("base64") });
      if (audio) send({ type: "response.audio_transcript.done", item_id: tr, transcript: "Hello there, my friend. How are you?" });
      else send({ type: "response.text.done", item_id: tr, text: "Hello there, my friend. How are you?" });
    });
  });
  return wss;
}

/** ElevenLabs: HTTP /v1/text-to-speech/:voice/stream и WebSocket /v1/text-to-dialogue/stream-input. */
export function startMockEleven(port, log = {}) {
  log.http = [];
  log.ws = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", async () => {
      const m = req.url.match(/^\/v1\/text-to-speech\/([^/]+)\/stream\?output_format=pcm_24000$/);
      if (!m || req.method !== "POST") return res.writeHead(404).end();
      if (req.headers["xi-api-key"] !== "el-key") return res.writeHead(401).end(JSON.stringify({ detail: { message: "bad key" } }));
      const j = JSON.parse(body);
      log.http.push({ voice: m[1], ...j });
      res.writeHead(200, { "content-type": "audio/pcm" });
      // Отдаём нечётными кусками, чтобы проверить склейку байтов.
      const pcm = tone(300);
      res.write(pcm.subarray(0, 1001));
      await new Promise((r) => setTimeout(r, 30));
      res.end(pcm.subarray(1001));
    });
  });
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (req, sock, head) => {
    if (!req.url.startsWith("/v1/text-to-dialogue/stream-input")) return sock.destroy();
    wss.handleUpgrade(req, sock, head, (ws) => {
      let hello = null;
      ws.on("message", (raw) => {
        const v = JSON.parse(raw.toString());
        if (!hello) {
          hello = v;
          if (v.xi_api_key !== "el-key") ws.send(JSON.stringify({ error: "auth", message: "bad key" }));
          return;
        }
        if (v.inputs) {
          log.ws.push(v.inputs[0].text);
          const pcm = tone(300);
          ws.send(JSON.stringify({ audio: pcm.subarray(0, 3001).toString("base64") }));
          ws.send(JSON.stringify({ audio: pcm.subarray(3001).toString("base64") }));
        }
        if (v.close_socket) {
          ws.send(JSON.stringify({ is_final: true }));
          ws.close();
        }
      });
    });
  });
  server.listen(port);
  return server;
}
