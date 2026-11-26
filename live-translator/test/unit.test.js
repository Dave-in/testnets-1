import { test } from "node:test";
import assert from "node:assert/strict";
import { buildSessionUpdate, classify, qwenUrl } from "../lib/qwen.js";
import { SentenceFeeder } from "../lib/sentences.js";
import { ElevenLabsSpeaker } from "../lib/elevenlabs.js";
import { startMockEleven } from "./mocks.mjs";

test("session.update: клон голоса на лету", () => {
  const s = buildSessionUpdate({ source: "ru", target: "ur", audio: true, voiceMode: "once" }).session;
  assert.deepEqual(s.modalities, ["text", "audio"]);
  assert.equal(s.voice, "default");
  assert.equal(s.enable_voice_clone, true);
  assert.deepEqual(s.voice_clone_options, { frequency: "once" });
  assert.equal(s.translation.language, "ur");
  assert.equal(s.input_audio_transcription.language, "ru");
  assert.equal(s.sample_rate, 16000);
});

test("session.update: готовый клон и готовый голос", () => {
  const f = buildSessionUpdate({ source: "ru", target: "en", audio: true, voiceMode: "fixed", voice: "qwen-tts-vc-x" }).session;
  assert.equal(f.voice, "qwen-tts-vc-x");
  assert.deepEqual(f.voice_clone_options, { frequency: "never" });
  const p = buildSessionUpdate({ source: "ru", target: "es", audio: true, voiceMode: "preset", voice: "Harvey" }).session;
  assert.equal(p.voice, "Harvey");
  assert.equal(p.enable_voice_clone, false);
});

test("session.update: только текст для ElevenLabs", () => {
  const s = buildSessionUpdate({ source: "ru", target: "en", audio: false }).session;
  assert.deepEqual(s.modalities, ["text"]);
  assert.equal(s.voice, undefined);
  assert.equal(s.enable_voice_clone, undefined);
});

test("адрес Qwen", () => {
  assert.equal(qwenUrl({ model: "m", region: "intl" }), "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=m");
  assert.equal(qwenUrl({ model: "m", region: "cn" }), "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=m");
  assert.equal(
    qwenUrl({ model: "m", apiHost: "https://ws-abc.ap-southeast-1.maas.aliyuncs.com/" }),
    "wss://ws-abc.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=m"
  );
});

test("classify", () => {
  assert.deepEqual(classify({ type: "response.audio_transcript.done", item_id: "a", transcript: "Hi." }), { kind: "target", id: "a", text: "Hi.", stash: "", final: true });
  assert.deepEqual(classify({ type: "conversation.item.created", previous_item_id: "s", item: { id: "t" } }), { kind: "link", id: "t", sourceId: "s" });
  assert.equal(classify({ type: "conversation.item.created", previous_item_id: null, item: { id: "s" } }), null);
  assert.equal(classify({ type: "error", error: { code: "InvalidApiKey", message: "x" } }).kind, "error");
});

test("SentenceFeeder: озвучивает целые предложения сразу, остаток — в конце", () => {
  const said = [];
  const f = new SentenceFeeder((t) => said.push(t));
  f.update("a", "Hello there, my", false);
  assert.deepEqual(said, []);
  f.update("a", "Hello there, my friend. How", false);
  assert.deepEqual(said, ["Hello there, my friend."]);
  f.update("a", "Hello there, my friend. How are you", false);
  f.update("a", "Hello there, my friend. How are you?", true);
  assert.deepEqual(said, ["Hello there, my friend.", "How are you?"]);
  f.update("a", "late duplicate", true);
  assert.equal(said.length, 2);
});

test("SentenceFeeder: короткое предложение ждёт следующего, урду с ۔", () => {
  const said = [];
  const f = new SentenceFeeder((t) => said.push(t));
  f.update("b", "Hi. How", false);
  assert.deepEqual(said, []);
  f.update("b", "Hi. How are you doing? I", false);
  assert.deepEqual(said, ["Hi. How are you doing?"]);
  const u = [];
  const g = new SentenceFeeder((t) => u.push(t));
  g.update("c", "میں آپ سے مل کر خوش ہوں۔ آپ", false);
  assert.deepEqual(u, ["میں آپ سے مل کر خوش ہوں۔"]);
});

test("SentenceFeeder: если текст переписан, ждём финал", () => {
  const said = [];
  const f = new SentenceFeeder((t) => said.push(t));
  f.update("d", "Good morning everyone. We", false);
  f.update("d", "Good day to all. We will", false);
  assert.deepEqual(said, ["Good morning everyone."]);
  f.update("d", "Good day to all. We will start now.", true);
  assert.equal(said.length, 2);
});

async function speakerRun(model) {
  const log = {};
  const srv = startMockEleven(0, log);
  await new Promise((r) => srv.once("listening", r));
  const port = srv.address().port;
  const chunks = [];
  const errors = [];
  const sp = new ElevenLabsSpeaker({
    apiKey: "el-key",
    voiceId: "voice123",
    model,
    language: "en",
    httpBase: `http://127.0.0.1:${port}`,
    wsBase: `ws://127.0.0.1:${port}`,
    onAudio: (b) => chunks.push(b),
    onError: (e) => errors.push(e),
  });
  sp.speak("First sentence.");
  sp.speak("Second sentence.");
  await new Promise((r) => setTimeout(r, 600));
  sp.close();
  srv.closeAllConnections();
  srv.close();
  return { log, chunks, errors };
}

test("ElevenLabs HTTP: звук по порядку, байты выровнены", async () => {
  const { log, chunks, errors } = await speakerRun("eleven_flash_v2_5");
  assert.deepEqual(errors, []);
  assert.equal(log.http.length, 2);
  assert.equal(log.http[0].language_code, "en");
  assert.equal(log.http[1].previous_text, "First sentence.");
  const total = chunks.reduce((n, c) => n + c.length, 0);
  assert.equal(total, 2 * 24000 * 0.3 * 2);
  assert.ok(chunks.every((c) => c.length % 2 === 0));
});

test("ElevenLabs v4 turbo через WebSocket", async () => {
  const { log, chunks, errors } = await speakerRun("eleven_v4_turbo");
  assert.deepEqual(errors, []);
  assert.deepEqual(log.ws, ["First sentence.", "Second sentence."]);
  const total = chunks.reduce((n, c) => n + c.length, 0);
  assert.equal(total, 2 * 24000 * 0.3 * 2);
});
