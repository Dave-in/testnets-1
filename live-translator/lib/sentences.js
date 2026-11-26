// Нарезает поток перевода от Qwen на предложения, чтобы ElevenLabs начинал говорить,
// не дожидаясь конца всей фразы.
//
// Qwen присылает для каждого элемента перевода снимки: text — подтверждённая часть (растёт),
// stash — черновик (может меняться). Озвучиваем только подтверждённый текст, и только целые
// предложения. Если подтверждённый текст вдруг перестал начинаться с уже озвученного,
// ждём финальный вариант.

// Конец предложения: . ! ? … и восклицание/вопрос в испанском/урду (۔ ؟), за которыми пробел.
const SENTENCE_END = /[.!?…۔؟](?=["»”')\]]*\s)/g;
const MIN_CHUNK = 12;

export class SentenceFeeder {
  /** @param {(text: string) => void} speak */
  constructor(speak) {
    this.speak = speak;
    this.items = new Map(); // id -> { spoken: string, diverged: boolean, done: boolean }
  }

  /** Подтверждённый текст элемента обновился (final=false) или пришёл окончательный (final=true). */
  update(id, text, final) {
    if (!id) return;
    let s = this.items.get(id);
    if (!s) {
      s = { spoken: "", diverged: false, done: false };
      this.items.set(id, s);
      if (this.items.size > 200) this.items.delete(this.items.keys().next().value);
    }
    if (s.done) return;

    if (!text.startsWith(s.spoken)) s.diverged = true;

    if (final) {
      s.done = true;
      let rest;
      if (!s.diverged) rest = text.slice(s.spoken.length);
      else rest = s.spoken.length === 0 ? text : text.slice(Math.min(s.spoken.length, text.length));
      rest = rest.trim();
      if (rest) this.speak(rest);
      s.spoken = text;
      return;
    }
    if (s.diverged) return;

    const tail = text.slice(s.spoken.length);
    let cut = -1;
    SENTENCE_END.lastIndex = 0;
    for (let m; (m = SENTENCE_END.exec(tail)); ) cut = m.index + 1;
    if (cut < 0) return;
    // Захватываем закрывающие кавычки/скобки после знака.
    while (cut < tail.length && /["»”')\]]/.test(tail[cut])) cut++;
    const chunk = tail.slice(0, cut).trim();
    if (chunk.length < MIN_CHUNK) return;
    s.spoken += tail.slice(0, cut);
    this.speak(chunk);
  }
}
