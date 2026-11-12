const LABELS = { rag_7b: 'RAG · 7B', rag_235b: 'RAG · 235B', ircot_7b: 'IRCoT · 7B', grasp_sft: 'GRASP · SFT', grasp_grpo: 'GRASP · GRPO' };
const ORDER = ['rag_7b', 'rag_235b', 'ircot_7b', 'grasp_sft', 'grasp_grpo'];
const state = { questions: [], selected: null, adapters: {}, stream: null, replaying: false };
const $ = (selector) => document.querySelector(selector);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function notice(message) { const box = $('#notice'); box.textContent = message || ''; box.hidden = !message; }
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}
const num = (value) => (value == null ? '–' : value.toFixed(2));
const triple = (t) => `<${t.head}; ${t.relation}; ${t.tail}>`;

function renderBoard(columns) {
  const board = $('#board');
  board.replaceChildren();
  for (const name of ORDER.filter((c) => columns.includes(c))) {
    const node = $('#column-template').content.firstElementChild.cloneNode(true);
    node.dataset.column = name;
    node.querySelector('h2').textContent = LABELS[name];
    if (name.startsWith('grasp_')) node.querySelector('.checkpoint').textContent = state.adapters[name.slice(6)]?.label || '';
    if (name === 'grasp_grpo') node.classList.add('pinned');
    board.append(node);
  }
  for (const item of document.querySelectorAll('#gold-docs li')) { item.classList.remove('found'); item.querySelector('.by').textContent = ''; }
}
const column = (name) => document.querySelector(`.column[data-column="${name}"]`);

function markGold(docId, columnName) {
  for (const item of document.querySelectorAll('#gold-docs li')) {
    if (item.dataset.doc !== String(docId)) continue;
    item.classList.add('found');
    const by = item.querySelector('.by');
    const names = new Set(by.textContent ? by.textContent.replace('tìm thấy bởi: ', '').split(', ') : []);
    names.add(LABELS[columnName]);
    by.textContent = `tìm thấy bởi: ${[...names].join(', ')}`;
  }
}

function openDoc(doc) {
  $('#doc-title').textContent = doc.title || doc.doc_id;
  $('#doc-meta').textContent = `${doc.doc_id}${doc.score != null ? ` · score ${doc.score.toFixed(3)}` : ''}${doc.gold ? ' · ★ tài liệu gold' : ''}`;
  $('#doc-text').textContent = doc.text || '';
  $('#doc-dialog').showModal();
}

function docList(docs, columnName) {
  const list = el('ol', 'docs');
  for (const doc of docs) {
    const item = el('li', doc.gold ? 'doc gold' : 'doc');
    const title = el('button', 'doc-title', `${doc.gold ? '★ ' : ''}${doc.title || doc.doc_id}`);
    title.onclick = () => openDoc(doc);
    item.append(title, el('span', 'snippet', (doc.text || '').slice(0, 200)));
    list.append(item);
    if (doc.gold) markGold(doc.doc_id, columnName);
  }
  return list;
}

function stepCard(type, data, columnName, t) {
  const card = el('details', `event ${type}`);
  card.open = type !== 'retrieve' || (data.step ?? 0) > 0;
  const summary = el('summary');
  const body = el('div', 'body');
  const step = data.step ? ` · bước ${data.step}` : '';
  if (type === 'retrieve') {
    summary.textContent = `Truy xuất${step} (${(data.docs || []).length} tài liệu)`;
    body.append(el('div', 'query', `Query: ${data.query}`), docList(data.docs || [], columnName));
  } else if (type === 'thought') {
    summary.textContent = `Suy nghĩ${step}`;
    body.append(el('p', null, data.text));
  } else if (type === 'rd') {
    summary.textContent = `RD${step}: ${data.retrieve ? 'cần truy xuất' : 'không truy xuất'}`;
  } else if (type === 'qf') {
    summary.textContent = `QF${step}`;
    body.append(el('div', 'query', `Query: ${data.query}`));
    if (data.chain && data.chain.length) body.append(el('div', 'muted', `Chuỗi: ${data.chain.map(triple).join(' → ')}`));
  } else if (type === 'ki') {
    summary.textContent = `KI${step}`;
    if (!data.candidates || !data.candidates.length) {
      body.append(el('div', 'muted', 'Không có triple ứng viên (tài liệu chưa có trong cache triple)'));
    } else {
      body.append(el('div', 'triple', triple(data.candidates[data.chosen ?? 0])));
      body.append(el('div', 'muted', data.added_doc ? `Thêm tài liệu ${data.added_doc} vào ngữ cảnh` : 'Tài liệu đã có trong ngữ cảnh'));
    }
  } else if (type === 'answer') {
    summary.textContent = 'Trả lời';
    body.append(el('p', 'answer-text', data.text || '(trống)'));
  } else if (type === 'error') {
    summary.textContent = 'Lỗi';
    body.append(el('p', null, data.message));
  } else {
    summary.textContent = type;
  }
  summary.append(el('span', 'time', `${t.toFixed(1)} s`));
  if (data.valid === false) summary.append(el('span', 'badge warn', '⚠ sai format'));
  if (data.raw) {
    const raw = el('details', 'raw');
    raw.append(el('summary', null, 'raw'), el('pre', null, data.raw));
    body.append(raw);
  }
  card.append(summary, body);
  return card;
}

function renderMetrics(host, d) {
  const metrics = host.querySelector('.metrics');
  metrics.replaceChildren();
  const add = (text, good) => metrics.append(el('span', good == null ? 'metric' : good ? 'metric good' : 'metric bad', text));
  if (d.f1 != null) {
    add(`F1 ${num(d.f1)}`, d.f1 >= 1 ? true : d.f1 <= 0 ? false : null);
    add(`EM ${num(d.em)}`, d.em >= 1);
  }
  if (d.recall != null) add(`Recall ${num(d.recall)}`, d.recall >= 1 ? true : d.recall <= 0 ? false : null);
  add(`${d.seconds.toFixed(1)} s`);
  add(`${d.retrievals} lần truy xuất`);
}

function applyEvent(event) {
  const host = column(event.column);
  if (!host) return;
  const data = event.data || {};
  if (event.type === 'done') { renderMetrics(host, data); return; }
  host.querySelector('.trace').append(stepCard(event.type, data, event.column, event.t || 0));
  if (event.type === 'answer') host.querySelector('.answer').textContent = data.text || '(trống)';
  if (event.type === 'error') {
    host.classList.add('failed');
    host.querySelector('.metrics').textContent = 'Lỗi';
  }
}

function showQuestion(question) {
  const scored = question && question.gold_answers && question.gold_answers.length;
  $('#question-info').hidden = !scored;
  $('#free-note').hidden = !!scored;
  if (!scored) return;
  $('#question-text').textContent = question.question;
  $('#gold-answer').textContent = question.gold_answers.join(' / ');
  $('#gold-docs').replaceChildren(...(question.gold_doc_ids || []).map((id) => {
    const item = el('li', null);
    item.dataset.doc = String(id);
    item.append(el('span', 'gold-id', `★ ${id}`), el('span', 'by muted', ''));
    return item;
  }));
}

function choose(question) {
  state.selected = question;
  $('#free-question').value = '';
  $('#eval-search').value = question.question;
  showQuestion(question);
}

async function loadQuestions() {
  const params = new URLSearchParams({ dataset: $('#dataset').value });
  if ($('#showcase').checked) params.set('showcase', '1');
  state.questions = await api(`/api/eval?${params}`);
  $('#eval-list').replaceChildren(...state.questions.map((q) => { const option = document.createElement('option'); option.value = q.question; return option; }));
}

async function ask() {
  notice('');
  const free = $('#free-question').value.trim();
  const columns = [...document.querySelectorAll('#columns input:checked')].map((x) => x.value);
  if (!columns.length) { notice('Chọn ít nhất một phương pháp.'); return; }
  const body = free ? { question: free, corpus: $('#free-corpus').value, columns } : state.selected ? { eval_id: state.selected.id, columns } : null;
  if (!body) { notice('Chọn một câu trong tập test hoặc gõ câu hỏi tự do.'); return; }
  if (free) showQuestion(null);
  if (state.stream) state.stream.close();
  renderBoard(columns);
  const { run_id } = await api('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const stream = new EventSource(`/api/stream/${run_id}`);
  state.stream = stream;
  stream.onmessage = (e) => applyEvent(JSON.parse(e.data));
  stream.addEventListener('close', () => stream.close());
  stream.onerror = () => { if (stream.readyState === EventSource.CLOSED) notice('Mất kết nối tới server.'); stream.close(); };
}

async function health() {
  const h = await api('/api/health');
  state.adapters = h.adapters || {};
  const off = Object.entries(h.columns).filter(([, v]) => !v.enabled);
  $('#health').textContent = off.length ? `Không dùng được: ${off.map(([k]) => LABELS[k]).join(', ')}` : 'Mọi cột sẵn sàng';
  $('#health').title = off.map(([k, v]) => `${LABELS[k]}: ${v.reason}`).join('\n');
  $('#health').classList.toggle('bad', off.length > 0);
  for (const [name, v] of Object.entries(h.columns)) {
    const box = document.querySelector(`#columns input[value="${name}"]`);
    if (box && !v.enabled) { box.checked = false; box.disabled = true; box.parentElement.title = v.reason || ''; }
  }
}

async function replay(row) {
  const events = await api(`/api/replay/${encodeURIComponent(row.file)}`);
  $('#history-dialog').close();
  if (state.stream) state.stream.close();
  showQuestion(row.request.example || null);
  renderBoard([...new Set(events.map((x) => x.column))]);
  const speed = Number($('#replay-speed').value);
  let previous = 0;
  for (const event of events) {
    await new Promise((resolve) => setTimeout(resolve, Math.max(0, ((event.t - previous) * 1000) / speed)));
    applyEvent(event);
    previous = event.t;
  }
}

async function showHistory() {
  const rows = await api('/api/history');
  $('#history-list').replaceChildren(...rows.map((row) => {
    const item = el('div', 'run');
    const when = row.created_at ? new Date(row.created_at * 1000).toLocaleString('vi-VN') : '';
    item.append(el('span', null, `${row.request.prerun ? '★ ' : ''}${row.request.example?.question || row.file}`), el('span', 'muted', when));
    const button = el('button', null, 'Phát lại');
    button.onclick = () => replay(row).catch((e) => notice(e.message));
    item.append(button);
    return item;
  }));
  $('#history-dialog').showModal();
}

async function prerun() {
  try { await api('/api/prerun', { method: 'POST' }); } catch (e) { notice(e.message); }
  const poll = async () => {
    const s = await api('/api/prerun');
    $('#prerun-status').textContent = s.error ? `Lỗi: ${s.error}` : s.running ? `Đang chạy ${s.done}/${s.total}…` : `Xong ${s.done}/${s.total}`;
    if (s.running) setTimeout(poll, 3000); else showHistory().catch((e) => notice(e.message));
  };
  poll();
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('grasp-theme', theme); } catch (e) { /* storage unavailable */ }
}

$('#dataset').onchange = () => loadQuestions().catch((e) => notice(e.message));
$('#showcase').onchange = () => loadQuestions().catch((e) => notice(e.message));
$('#eval-search').onchange = () => { const found = state.questions.find((q) => q.question === $('#eval-search').value); if (found) choose(found); };
$('#random').onclick = () => { if (state.questions.length) choose(state.questions[Math.floor(Math.random() * state.questions.length)]); };
$('#ask').onclick = () => ask().catch((e) => notice(e.message));
$('#history').onclick = () => showHistory().catch((e) => notice(e.message));
$('#close-history').onclick = () => $('#history-dialog').close();
$('#close-doc').onclick = () => $('#doc-dialog').close();
$('#prerun').onclick = () => prerun();
$('#theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
try { const saved = localStorage.getItem('grasp-theme'); if (saved) document.documentElement.dataset.theme = saved; } catch (e) { /* storage unavailable */ }
loadQuestions().catch((e) => notice(e.message));
health().catch((e) => { $('#health').textContent = `Không kiểm tra được dịch vụ: ${e.message}`; });
