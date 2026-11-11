"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("query-form");
const input = byId("query");
const submit = byId("submit");
const useQwen = byId("use-qwen");
const feedback = byId("feedback");
const modeButtons = [...document.querySelectorAll("[data-mode]")];
const examples = [...document.querySelectorAll("[data-example-mode]")];
let mode = "qa";

function setMode(nextMode) {
  mode = nextMode;
  const searching = ["search", "corpus", "community"].includes(mode);
  modeButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
  byId("query-label").textContent = mode === "community" ? "人物或实体名称" : searching ? "原文关键词" : "你的问题";
  input.placeholder = mode === "community" ? "例如：云长" : mode === "corpus" ? "例如：第98回 王双 魏延" : searching ? "例如：刘备 既结为兄弟" : "例如：刘备和关羽在桃园结义阶段是什么关系？";
  byId("query-hint").textContent = mode === "corpus"
    ? "支持按回检索。用空格分开多个关键词，引用保留底本字形。"
    : mode === "community" ? "输入名称或别名，查看所在社区和关联证据。"
    : searching ? "用空格分开多个关键词，原文需同时包含这些词。" : "可查人物关系、事件和发生阶段，例如“云长何时斩杀华雄？”。";
  byId("qwen-option").hidden = searching;
  submit.textContent = searching ? "检索" : "提问";
  if (mode === "completion") {
    input.placeholder = "例如：第6回，徐荣追击了谁？";
    byId("query-hint").textContent = "指定回数、人物和动作。原文证据与待核验候选会分开显示。";
    byId("qwen-option").hidden = true;
    submit.textContent = "查看候选";
  }
}

function makeSource(source) {
  const item = document.createElement("li");
  item.className = "source";
  const fact = document.createElement("p");
  fact.textContent = source.time ? `${source.fact}（${source.time}）` : source.fact;
  const quote = document.createElement("blockquote");
  quote.textContent = source.quote;
  const citation = document.createElement("small");
  citation.textContent = `《三国演义》${source.chapter} · ${source.chunk} · ${source.id}`;
  item.append(fact, quote, citation);
  return item;
}

function renderResult(result) {
  byId("answer-title").textContent = result.method.endsWith("检索") ? "检索结果" : "回答";
  const method = [result.method];
  if (result.query_parser === "bert") method.push("本地 BERT");
  if (result.retrieval_backend === "neo4j") method.push("Neo4j");
  byId("method").textContent = method.join(" · ");
  byId("answer").textContent = result.answer;
  byId("notice").textContent = result.notice;
  byId("notice").hidden = !result.notice;
  byId("sources").replaceChildren(...result.sources.map(makeSource));
  byId("source-title").textContent = result.evidence_status === "clue" ? "相关原文线索" : "依据与出处";
  byId("source-section").hidden = !result.sources.length;
  renderCompletion(result.completion);
  byId("result").hidden = false;
  feedback.textContent = result.evidence_status === "clue" ? "找到相关段落，结论仍需核对。"
    : result.sources.length ? `返回 ${result.sources.length} 条有出处的记录。` : "没有可引用的记录。";
}

function renderCompletion(completion) {
  byId("completion-section").hidden = !completion;
  if (!completion) return;
  byId("completion-note").textContent = completion.message ||
    `${completion.model} · 种子 ${completion.seed} · ${completion.time} · 模型关系“${completion.model_relation}”。未找到直接证据的候选仅供核验。`;
  const rows = (completion.candidates || []).map((candidate) => {
    const item = document.createElement("li");
    item.className = "source";
    const state = candidate.evidence_ids.length ? `有直接证据 · ${candidate.evidence_ids.join("、")}` : "待核验";
    item.textContent = `第 ${candidate.rank} 名 · ${candidate.entity} · ${state}`;
    return item;
  });
  byId("completion-candidates").replaceChildren(...rows);
}

function setBusy(busy) {
  [submit, ...modeButtons, ...examples].forEach((button) => { button.disabled = busy; });
  form.setAttribute("aria-busy", String(busy));
}

async function handleSubmit(event) {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) {
    feedback.textContent = "请输入问题或关键词。";
    input.focus();
    return;
  }
  setBusy(true);
  feedback.textContent = "正在查询…";
  byId("result").hidden = true;
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode, use_qwen: mode === "qa" && useQwen.checked }),
    });
    const result = await response.json();
    if (!response.ok) {
      feedback.textContent = result.error || "查询失败，请重试。";
      return;
    }
    renderResult(result);
    if (mode === "completion") byId("result").scrollIntoView({ block: "start" });
  } catch {
    feedback.textContent = "无法连接本地问答服务，请确认服务仍在运行。";
  } finally {
    setBusy(false);
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const status = await response.json();
    const summary = status.book_chapters
      ? `${status.fact_count} 条事实 · 全文 ${status.book_chapters} 回`
      : `${status.fact_count} 条带出处的事实`;
    byId("connection").textContent = status.bert_enabled ? `${summary} · 本地 BERT 已启用` : summary;
    useQwen.disabled = !status.qwen_enabled;
    byId("live-completion-status").textContent = status.completion_enabled
      ? "实时补全已启用。点击按钮运行 RE-GCN 并跳到原文核验结果。"
      : "当前服务尚未启用实时补全。按软件运行说明用 --completion 启动后可用。";
    byId("qwen-hint").textContent = status.qwen_enabled
      ? "勾选后会把问题和匹配的图谱证据或原文段落发给百炼。"
      : "未配置千问，可直接用本地问答。";
  } catch {
    byId("connection").textContent = "服务状态未知";
  }
}

modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
examples.forEach((button) => button.addEventListener("click", () => {
  setMode(button.dataset.exampleMode);
  input.value = button.dataset.query;
  form.requestSubmit();
}));
form.addEventListener("submit", handleSubmit);
loadStatus();

function renderPrediction(data, index) {
  const selected = data.cases[index];
  const reveal = byId("reveal-answer");
  byId("prediction-answer").hidden = true;
  byId("prediction-answer").replaceChildren();
  reveal.setAttribute("aria-expanded", "false");
  reveal.disabled = !data.review_available;
  reveal.textContent = data.review_available ? "揭晓数据集答案" : "答案文件暂不可用，仍可查看候选";
  reveal.onclick = () => togglePredictionAnswer(selected);
  byId("prediction-question").textContent = `${selected.head} —${selected.relation}→ ______（${selected.time}）`;
  const cards = Object.entries({ tirgn: "TiRGN", regcn: "RE-GCN" }).map(([key, label]) => {
    const card = document.createElement("section");
    card.className = "prediction-model";
    const title = document.createElement("h3");
    title.textContent = label;
    const provenance = document.createElement("p");
    provenance.className = "hint";
    const seed = data.models[key].seed;
    provenance.textContent = `${seed ? `种子 ${seed} · ` : ""}第 ${data.models[key].best_epoch} 轮的最佳验证权重 · 前 ${data.top_k} 名`;
    const list = document.createElement("ol");
    selected.predictions[key].forEach((candidate) => {
      const item = document.createElement("li");
      item.textContent = candidate.entity;
      item.dataset.entityId = String(candidate.entity_id);
      list.append(item);
    });
    card.append(title, provenance, list);
    return card;
  });
  byId("prediction-models").replaceChildren(...cards);
}

function togglePredictionAnswer(selected) {
  const target = byId("prediction-answer");
  const reveal = byId("reveal-answer");
  if (!target.hidden) {
    target.hidden = true;
    reveal.textContent = "揭晓数据集答案";
    reveal.setAttribute("aria-expanded", "false");
    document.querySelectorAll(".prediction-model .matched-answer").forEach((item) => item.classList.remove("matched-answer"));
    return;
  }
  const answers = new Set(selected.answers.map((item) => item.entity_id));
  const explanation = document.createElement("p");
  explanation.textContent = `数据集记录的答案：${selected.answers.map((item) => item.entity).join("、")}。`;
  const results = Object.entries({ tirgn: "TiRGN", regcn: "RE-GCN" }).map(([key, label]) => {
    const line = document.createElement("p");
    const matches = selected.predictions[key].filter((item) => answers.has(item.entity_id));
    line.textContent = matches.length
      ? `${label}：${matches.map((item) => `${item.entity}排第 ${item.rank} 名`).join("；")}。`
      : `${label}：前 ${selected.predictions[key].length} 名没有命中已记录的答案。`;
    return line;
  });
  const note = document.createElement("small");
  note.textContent = "同一道题可能有多个答案。这些是验证集中的标注，仍需原文核对；未命中标注不自动证明其他候选为假。揭晓答案不会改变模型排名。";
  target.replaceChildren(explanation, ...results, note);
  target.hidden = false;
  document.querySelectorAll(".prediction-model li").forEach((item) => {
    item.classList.toggle("matched-answer", answers.has(Number(item.dataset.entityId)));
  });
  reveal.textContent = "收起数据集答案";
  reveal.setAttribute("aria-expanded", "true");
}

async function loadPredictions() {
  const status = byId("prediction-status");
  try {
    const response = await fetch("/api/predictions");
    if (!response.ok) throw new Error("Prediction request failed");
    const data = await response.json();
    if (!data.available) {
      status.textContent = data.message;
      return;
    }
    const select = byId("prediction-case");
    data.cases.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${item.head} · ${item.relation} · ${item.time}`;
      select.append(option);
    });
    select.addEventListener("change", () => renderPrediction(data, Number(select.value)));
    renderPrediction(data, 0);
    byId("prediction-content").hidden = false;
    const version = data.data_version ? "v2 全文复核版" : "先导数据版";
    status.textContent = `${version} · ${data.cases.length} 个验证案例`;
  } catch {
    status.textContent = "预测案例暂不可用，请确认本地服务正常。事实问答可单独使用。";
  }
}

loadPredictions();
