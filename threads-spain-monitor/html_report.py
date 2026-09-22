"""Одностраничный HTML-отчёт: посты с лайками/комментариями, клик по комментариям
раскрывает самые залайканные ответы. Файл самодостаточный — открывается двойным кликом."""
from __future__ import annotations

import json
from collections import defaultdict

MAX_REPLIES_PER_POST = 30


def _query_summary(posts: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {"posts": 0, "likes": 0, "comments": 0, "top": None})
    for p in posts:
        for q in p.get("queries") or ["(без запроса)"]:
            a = agg[q]
            a["posts"] += 1
            a["likes"] += p.get("likes") or 0
            a["comments"] += p.get("replies") or 0
            if a["top"] is None or (p.get("likes") or 0) > (a["top"].get("likes") or 0):
                a["top"] = {"id": p["id"], "text": p["text"][:140], "likes": p.get("likes")}
    rows = [{"query": q, **v} for q, v in agg.items()]
    rows.sort(key=lambda r: (r["likes"] + 2 * r["comments"], r["posts"]), reverse=True)
    return rows


def render_html(res: dict, replies_by_post: dict[str, list[dict]]) -> str:
    posts = res.get("all_posts") or res.get("top_posts") or []
    slim = []
    for p in posts:
        reps = replies_by_post.get(p["id"], [])[:MAX_REPLIES_PER_POST]
        slim.append({
            "id": p["id"], "text": p["text"], "user": p.get("username"), "url": p.get("permalink"),
            "ts": p.get("timestamp"), "likes": p.get("likes"), "comments": p.get("replies"),
            "reposts": p.get("reposts"), "score": p.get("score"), "queries": p.get("queries") or [],
            "topics": list(p.get("topics") or {}), "intents": list(p.get("intents") or {}),
            "locs": list(p.get("locations") or {}),
            "repl": [{"text": r["text"], "user": r.get("username"), "url": r.get("permalink"),
                      "ts": r.get("timestamp"), "likes": r.get("like_count"),
                      "comments": r.get("reply_count")} for r in reps],
        })
    data = {
        "generated": res.get("generated_at"), "days": res.get("window_days"),
        "intents": res.get("intents", {}), "locations": res.get("locations", {}),
        "trends": [{k: t[k] for k in ("topic", "posts", "prev_posts", "growth", "status", "heat", "intents", "keywords")}
                   for t in res.get("trends", [])],
        "polls": res.get("polls", []), "queries": _query_summary(posts), "posts": slim,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("/*__DATA__*/null", payload)


TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Тренды Threads · Коста Бланка</title>
<style>
:root{
  --bg:#f6f5f2; --card:#ffffff; --ink:#1c1b19; --muted:#6b6862; --line:#e4e1da;
  --accent:#c2410c; --accent-soft:#fdeee4; --like:#be123c; --good:#15803d; --bad:#b91c1c; --chip:#efece6;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#141412; --card:#1d1c1a; --ink:#ecebe7; --muted:#9b978f; --line:#2f2d2a;
         --accent:#fb923c; --accent-soft:#3a2416; --like:#fb7185; --good:#4ade80; --bad:#f87171; --chip:#2a2926; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px} .sub{color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kpi b.word{font-size:16px;line-height:1.3;padding:3px 0} .kpi span{color:var(--muted);font-size:12px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 16px;overflow-x:auto}
.tab{background:none;border:0;padding:10px 14px;font:inherit;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.tab.on{color:var(--ink);border-color:var(--accent);font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.controls input,.controls select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.controls input{flex:1;min-width:180px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.row{display:grid;grid-template-columns:34px 1fr 78px 92px;gap:12px;padding:12px 14px;border-top:1px solid var(--line);align-items:start}
.row:first-child{border-top:0}
.row.head{font-size:12px;color:var(--muted);padding:8px 14px;background:var(--chip);border-top:0}
.row.head button{all:unset;cursor:pointer} .row.head button.on{color:var(--ink);font-weight:600}
.num{font-variant-numeric:tabular-nums;text-align:right}
.rank{color:var(--muted);font-variant-numeric:tabular-nums}
.text{white-space:pre-wrap;word-break:break-word}
.text.clamp{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.meta{color:var(--muted);font-size:12px;margin-top:4px}
.meta a{color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.chip{background:var(--chip);border-radius:999px;padding:1px 8px;font-size:11px;color:var(--muted)}
.chip.neg{color:var(--bad)} .chip.q{color:var(--accent)}
.likes{color:var(--like);font-weight:600}
.cbtn{font:inherit;font-weight:600;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:4px 8px;cursor:pointer;width:100%;text-align:right;font-variant-numeric:tabular-nums}
.cbtn:hover,.cbtn.open{border-color:var(--accent);background:var(--accent-soft)}
.cbtn:disabled{opacity:.5;cursor:default;background:none}
.replies{grid-column:1/-1;background:var(--bg);border-radius:10px;padding:10px 12px;margin:2px 0 4px}
.replies h4{margin:0 0 8px;font-size:13px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.replies label{font-weight:400;color:var(--muted);cursor:pointer}
.rep{display:grid;grid-template-columns:1fr 64px 64px;gap:10px;padding:8px 0;border-top:1px dashed var(--line)}
.rep:first-of-type{border-top:0}
.badge{display:inline-block;background:var(--accent-soft);color:var(--accent);border-radius:6px;padding:0 6px;font-size:11px;margin-left:4px}
.empty{padding:24px;text-align:center;color:var(--muted)}
table{width:100%;border-collapse:collapse}
th,td{padding:10px 12px;border-top:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--muted);background:var(--chip);border-top:0;font-weight:500}
tr.click{cursor:pointer} tr.click:hover td{background:var(--accent-soft)}
.up{color:var(--good)} .down{color:var(--bad)}
.polls{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:16px}
.poll{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.poll b{display:block;margin-bottom:8px} .poll li{margin:2px 0} .poll ul{margin:0;padding-left:18px}
h3{font-size:16px;margin:24px 0 8px}
.scroll{overflow-x:auto}
@media (max-width:640px){
  .row{grid-template-columns:1fr 64px 80px} .row .rank,.row.head>span:first-child{display:none}
  .controls select{flex:1 1 45%}
  .kpis{grid-template-columns:repeat(2,1fr)}
  .rep{grid-template-columns:1fr 52px 52px}
}
</style>
</head>
<body>
<div class="wrap">
  <h1>Тренды Threads: Испания · Аликанте · Коста Бланка</h1>
  <div class="sub" id="sub"></div>
  <div class="kpis" id="kpis"></div>
  <div class="tabs" role="tablist">
    <button class="tab on" data-tab="posts">Посты</button>
    <button class="tab" data-tab="queries">Запросы</button>
    <button class="tab" data-tab="topics">Темы и опросы</button>
  </div>
  <section id="tab-posts">
    <div class="controls">
      <input id="f-text" type="search" placeholder="Поиск по тексту…">
      <select id="f-query"><option value="">Все запросы</option></select>
      <select id="f-topic"><option value="">Все темы</option></select>
      <select id="f-intent"><option value="">Любая эмоция</option></select>
    </div>
    <div class="card" id="posts"></div>
  </section>
  <section id="tab-queries" hidden><div class="card scroll" id="queries"></div></section>
  <section id="tab-topics" hidden>
    <div class="card scroll" id="trends"></div>
    <h3>Идеи опросов</h3>
    <div class="polls" id="polls"></div>
  </section>
</div>
<script>
const D = /*__DATA__*/null;
const fmt = n => (n === null || n === undefined) ? "—" : Number(n).toLocaleString("ru-RU");
const day = s => s ? s.slice(0,10) : "";
function h(tag, attrs, ...kids){
  const el = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs||{})){
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false)
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
const state = { sort: "likes", open: new Set(), onlyAnswered: new Set() };

document.getElementById("sub").textContent =
  `Сформировано ${D.generated?.replace("T"," ").slice(0,16) || ""} · окно ${D.days} дн. · клик по числу комментариев открывает самые залайканные ответы`;

const sum = k => D.posts.reduce((a,p) => a + (p[k] || 0), 0);
const topTrend = D.trends[0];
document.getElementById("kpis").append(
  ...[["Постов", fmt(D.posts.length)], ["Лайков", fmt(sum("likes"))], ["Комментариев", fmt(sum("comments"))],
      ["Главная тема", topTrend ? topTrend.topic : "—"],
      ["Главная эмоция", Object.keys(D.intents)[0] || "—"]]
  .map(([l,v]) => h("div",{class:"kpi"}, h("b",{class: /\d/.test(v) ? "" : "word"},v), h("span",{},l))));

function fillSelect(id, values){ const s = document.getElementById(id);
  for (const v of values) s.append(h("option",{value:v},v)); s.addEventListener("change", renderPosts); }
fillSelect("f-query", D.queries.map(q => q.query));
fillSelect("f-topic", [...new Set(D.posts.flatMap(p => p.topics))].sort());
fillSelect("f-intent", Object.keys(D.intents));
document.getElementById("f-text").addEventListener("input", renderPosts);

function chips(p){
  return h("div",{class:"chips"},
    p.intents.map(i => h("span",{class:"chip" + (i==="негатив"?" neg":i==="вопрос"?" q":"")}, i)),
    p.topics.map(t => h("span",{class:"chip"}, t)),
    p.locs.filter(l => l !== "Испания").map(l => h("span",{class:"chip"}, "📍 " + l)));
}
function sortBtn(key, label){
  return h("button",{class: state.sort===key ? "on" : "", onclick:()=>{state.sort=key; renderPosts();}}, label + (state.sort===key?" ↓":""));
}
function renderReplies(p){
  const only = state.onlyAnswered.has(p.id);
  let list = [...p.repl].sort((a,b) => (b.likes||0)-(a.likes||0) || (b.comments||0)-(a.comments||0));
  if (only) list = list.filter(r => (r.comments||0) > 0);
  const box = h("div",{class:"replies"},
    h("h4",{}, `Топ ответов по лайкам (${p.repl.length} из ${fmt(p.comments)})`,
      h("label",{}, h("input",{type:"checkbox", checked: only, onchange:e=>{
          e.target.checked ? state.onlyAnswered.add(p.id) : state.onlyAnswered.delete(p.id); renderPosts(); }}),
        " только с ответами")));
  if (!list.length) box.append(h("div",{class:"meta"}, only ? "Нет ответов, на которые ответили." : "Ответы не собраны — запустите сбор с открытием постов."));
  for (const r of list) box.append(h("div",{class:"rep"},
    h("div",{}, h("div",{class:"text"}, r.text),
      h("div",{class:"meta"}, r.user ? "@"+r.user+" · " : "", day(r.ts),
        r.url ? [" · ", h("a",{href:r.url, target:"_blank", rel:"noopener"},"открыть")] : null,
        (r.comments||0) > 0 ? h("span",{class:"badge"}, "есть ответы") : null)),
    h("div",{class:"num likes", title:"лайки"}, "❤ " + fmt(r.likes)),
    h("div",{class:"num", title:"ответы на этот комментарий"}, "💬 " + fmt(r.comments))));
  return box;
}
function renderPosts(){
  const t = document.getElementById("f-text").value.trim().toLowerCase();
  const q = document.getElementById("f-query").value, tp = document.getElementById("f-topic").value,
        it = document.getElementById("f-intent").value;
  let rows = D.posts.filter(p => (!t || p.text.toLowerCase().includes(t)) && (!q || p.queries.includes(q))
    && (!tp || p.topics.includes(tp)) && (!it || p.intents.includes(it)));
  const key = { likes: p => p.likes||0, comments: p => p.comments||0, score: p => p.score||0, date: p => Date.parse(p.ts)||0 }[state.sort];
  rows.sort((a,b) => key(b) - key(a));
  const box = document.getElementById("posts"); box.replaceChildren();
  box.append(h("div",{class:"row head"}, h("span",{},"#"),
    h("span",{}, "Пост · ", sortBtn("score","по жару"), " · ", sortBtn("date","по дате")),
    h("span",{class:"num"}, sortBtn("likes","❤ Лайки")), h("span",{class:"num"}, sortBtn("comments","💬 Коммент."))));
  if (!rows.length){ box.append(h("div",{class:"empty"},"Ничего не найдено")); return; }
  rows.forEach((p,i) => {
    const open = state.open.has(p.id);
    const row = h("div",{class:"row"},
      h("span",{class:"rank"}, i+1),
      h("div",{}, h("div",{class:"text" + (open ? "" : " clamp")}, p.text),
        h("div",{class:"meta"}, p.user ? "@"+p.user+" · " : "", day(p.ts),
          p.url ? [" · ", h("a",{href:p.url, target:"_blank", rel:"noopener"},"открыть в Threads")] : null,
          p.queries.length ? " · запрос: " + p.queries.join(", ") : ""),
        chips(p)),
      h("div",{class:"num likes"}, fmt(p.likes)),
      h("div",{}, h("button",{class:"cbtn" + (open?" open":""), disabled: !(p.repl.length || p.comments),
        title:"показать топ ответов", onclick:()=>{ open ? state.open.delete(p.id) : state.open.add(p.id); renderPosts(); }},
        fmt(p.comments) + (open ? " ▴" : " ▾"))));
    if (open) row.append(renderReplies(p));
    box.append(row);
  });
}
function renderQueries(){
  const tb = h("tbody");
  for (const q of D.queries) tb.append(h("tr",{class:"click", title:"показать посты по запросу", onclick:()=>{
      document.getElementById("f-query").value = q.query; showTab("posts"); renderPosts(); }},
    h("td",{}, h("b",{}, q.query)), h("td",{class:"num"}, fmt(q.posts)),
    h("td",{class:"num likes"}, fmt(q.likes)), h("td",{class:"num"}, fmt(q.comments)),
    h("td",{class:"meta"}, q.top ? q.top.text : "")));
  document.getElementById("queries").append(h("table",{},
    h("thead",{}, h("tr",{}, ["Запрос","Постов","❤ Лайков","💬 Коммент.","Самый залайканный пост"].map(x => h("th",{},x)))), tb));
}
function renderTopics(){
  const tb = h("tbody");
  for (const t of D.trends){
    const g = t.growth === null ? h("span",{class:"up"},"новое") :
      h("span",{class: t.growth > 0 ? "up" : t.growth < 0 ? "down" : ""}, (t.growth>0?"+":"") + Math.round(t.growth*100) + "%");
    tb.append(h("tr",{class:"click", onclick:()=>{ document.getElementById("f-topic").value = t.topic; showTab("posts"); renderPosts(); }},
      h("td",{}, h("b",{}, t.topic)), h("td",{class:"num"}, t.posts), h("td",{class:"num"}, g),
      h("td",{class:"num"}, t.heat), h("td",{}, Object.entries(t.intents).slice(0,3).map(([k,v]) => `${k} ${v}`).join(", ")),
      h("td",{class:"meta"}, t.keywords.join(", "))));
  }
  document.getElementById("trends").append(h("table",{},
    h("thead",{}, h("tr",{}, ["Тема","Постов","Динамика","Жар","Эмоции","Слова"].map(x => h("th",{},x)))), tb));
  document.getElementById("polls").append(...D.polls.map(p => h("div",{class:"poll"},
    h("b",{}, p.question), h("ul",{}, p.options.map(o => h("li",{},o))), h("div",{class:"meta"}, p.topic + " · " + p.why))));
}
function showTab(name){
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  for (const s of ["posts","queries","topics"]) document.getElementById("tab-"+s).hidden = s !== name;
}
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
renderPosts(); renderQueries(); renderTopics();
</script>
</body>
</html>
"""
