// AudioWorklet: берёт звук микрофона на частоте устройства (обычно 48 кГц),
// пересчитывает в 16 кГц PCM16 mono и отдаёт кусками по 100 мс.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // дробная позиция следующего выходного отсчёта во входном потоке
    this.prev = 0; // последний отсчёт предыдущего блока (для интерполяции)
    this.out = new Int16Array(1600);
    this.n = 0;
    this.muted = false;
    this.port.onmessage = (e) => {
      if (e.data && typeof e.data.muted === "boolean") this.muted = e.data.muted;
    };
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const len = ch.length;
    let peak = 0;
    // Линейная интерполяция между отсчётами; индекс -1 — это this.prev.
    while (this.pos < len) {
      const i = Math.floor(this.pos);
      const f = this.pos - i;
      const a = i === 0 ? this.prev : ch[i - 1];
      const b = ch[i];
      let v = this.muted ? 0 : a + (b - a) * f;
      if (v > 1) v = 1;
      else if (v < -1) v = -1;
      const av = v < 0 ? -v : v;
      if (av > peak) peak = av;
      this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      if (this.n === this.out.length) {
        this.port.postMessage({ pcm: this.out.buffer, peak }, [this.out.buffer]);
        this.out = new Int16Array(1600);
        this.n = 0;
        peak = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= len;
    this.prev = ch[len - 1];
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
