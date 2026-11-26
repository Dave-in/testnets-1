// Сквозная проверка: настоящий браузер (поддельный микрофон) → server.js → поддельные Qwen/ElevenLabs.
// Запуск: npm run test:e2e   (нужен playwright с Chromium)
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { startMockQwen, startMockEleven } from "./mocks.mjs";

const require = createRequire(import.meta.url);
const { chromium, devices } = require("playwright");
const root = fileURLToPath(new URL("..", import.meta.url));
const shots = process.env.SHOTS_DIR;

const qlog = {};
const elog = {};
const qwen = startMockQwen(18911, qlog);
const eleven = startMockEleven(18912, elog);

const server = spawn(process.execPath, ["server.js"], {
  cwd: root,
  env: {
    ...process.env,
    PORT: "18910",
    APP_PASSWORD: "secret",
    DASHSCOPE_API_KEY: "qwen-key",
    QWEN_WS_URL: "ws://127.0.0.1:18911/api-ws/v1/realtime",
    ELEVENLABS_API_KEY: "el-key",
    ELEVENLABS_VOICE_ID: "myvoice",
    ELEVENLABS_MODEL: process.env.E2E_EL_MODEL || "eleven_v4_turbo",
    ELEVENLABS_HTTP_BASE: "http://127.0.0.1:18912",
    ELEVENLABS_WS_BASE: "ws://127.0.0.1:18912",
  },
  stdio: ["ignore", "inherit", "inherit"],
});
await new Promise((r) => setTimeout(r, 800));

const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});
let failed = false;
try {
  const context = await browser.newContext({ ...devices["Pixel 7"], permissions: ["microphone"] });
  const page = await context.newPage();
  page.on("pageerror", (e) => {
    console.error("pageerror:", e.message);
    failed = true;
  });
  let audioFrames = 0;
  page.on("websocket", (ws) => ws.on("framereceived", (f) => typeof f.payload !== "string" && audioFrames++));

  await page.goto("http://127.0.0.1:18910/");
  await page.locator("#targets button").first().waitFor();
  if (shots) await page.screenshot({ path: `${shots}/01-start.png` });

  // Без пароля приложение просит его ввести.
  await page.click("#micBtn");
  assert.equal(await page.locator("details.settings").getAttribute("open"), "");
  await page.fill("#password", "secret");
  await page.dispatchEvent("#password", "change");

  for (const mode of ["qwen", "elevenlabs"]) {
    audioFrames = 0;
    await page.check(`input[name="mode"][value="${mode}"]`);
    await page.click('#targets button[data-code="es"]');
    await page.click("#micBtn");
    await page.locator(".turn .dst").first().waitFor({ timeout: 10000 });
    await page.waitForFunction(() => document.querySelector(".turn .dst")?.textContent.includes("How are you?"), null, { timeout: 10000 });
    await page.waitForTimeout(1500);
    const turns = await page.locator(".turn").count();
    const src = await page.locator(".turn .src").first().textContent();
    const latency = await page.locator("#latency").textContent();
    console.log(`[${mode}] реплик: ${turns}, оригинал: «${src}», аудио-кадров: ${audioFrames}, ${latency}`);
    assert.ok(src.includes("Как дела?"));
    assert.ok(audioFrames > 0, "звук перевода не пришёл");
    if (shots) await page.screenshot({ path: `${shots}/02-${mode}.png` });
    await page.click("#micBtn");
    await page.waitForTimeout(300);
    assert.equal(await page.locator("#micBtn").getAttribute("aria-pressed"), "false");
    await page.click("#clearLog");
  }

  const [s1, s2] = qlog.sessions;
  assert.equal(qlog.auth[0], "Bearer qwen-key");
  assert.deepEqual(s1.modalities, ["text", "audio"]);
  assert.equal(s1.voice, "default");
  assert.equal(s1.translation.language, "es");
  assert.deepEqual(s2.modalities, ["text"]);
  const spoken = [...elog.ws, ...elog.http.map((h) => h.text)];
  console.log("ElevenLabs озвучил:", spoken);
  assert.ok(spoken.includes("Hello there, my friend."), "первое предложение не ушло в ElevenLabs заранее");
  assert.ok(spoken.includes("How are you?"));
  console.log(`Qwen получил ${qlog.appends} кусков звука по 100 мс`);
  console.log(failed ? "E2E: ОШИБКИ В СТРАНИЦЕ" : "E2E: OK");
} catch (e) {
  failed = true;
  console.error(e);
} finally {
  await browser.close();
  server.kill();
  qwen.close();
  eleven.closeAllConnections();
  eleven.close();
  process.exit(failed ? 1 : 0);
}
