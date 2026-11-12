const LABELS = { rag_7b: 'RAG · 7B', rag_235b: 'RAG · 235B', ircot_7b: 'IRCoT · 7B', grasp_sft: 'GRASP · SFT', grasp_grpo: 'GRASP · GRPO' };
const ORDER = ['rag_7b', 'rag_235b', 'ircot_7b', 'grasp_sft', 'grasp_grpo'];
const state = { questions: [], selected: null, adapters: {}, stream: null, generation: 0, titles: {}, goldCount: 0, seen: {} };
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
    if (name === 'rag_7b' || name === 'rag_235b') node.classList.add('secondary');
    board.append(node);
  }
  state.seen = {};
  renderSummary(columns);
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

function renderSummary(columns) {
  const body = $('#summary-body');
  body.replaceChildren();
  for (const name of ORDER.filter((c) => columns.includes(c))) {
    const row = el('tr');
    row.dataset.column = name;
    const label = el('td', 'method');
    label.append(el('strong', null, LABELS[name]));
    if (name.startsWith('grasp_')) label.append(el('small', 'muted', ` ${state.adapters[name.slice(6)]?.label || ''}`));
    row.append(label, el('td', 'verdict pending', '…'), el('td', 'answer-cell', ''), el('td', 'num searches', '–'), el('td', 'num ctx', '–'), el('td', 'num gold', '–'), el('td', 'num secs', '–'));
    body.append(row);
  }
  $('#summary').hidden = !columns.length;
}
const summaryRow = (name) => document.querySelector(`#summary-body tr[data-column="${name}"]`);

function verdict(d) {
  if (d.f1 == null) return ['–', 'neutral', 'Câu tự do: không có đáp án để chấm'];
  if (d.em >= 1) return ['✓ Đúng', 'good', `F1 ${num(d.f1)} · EM 1`];
  if (d.f1 > 0) return ['≈ Một phần', 'partial', `F1 ${num(d.f1)} · EM 0`];
  return ['✗ Sai', 'bad', `F1 ${num(d.f1)} · EM 0`];
}

function updateSummary(name, type, data) {
  const row = summaryRow(name);
  if (!row) return;
  if (type === 'answer') row.querySelector('.answer-cell').textContent = data.text || '(trống)';
  if (type === 'error') {
    const cell = row.querySelector('.verdict');
    cell.className = 'verdict bad';
    cell.textContent = 'Lỗi';
    cell.title = data.message || '';
  }
  if (type === 'done') {
    const [text, cls, title] = verdict(data);
    const cell = row.querySelector('.verdict');
    cell.className = `verdict ${cls}`;
    cell.textContent = text;
    cell.title = title;
    row.querySelector('.searches').textContent = data.retrievals ?? '–';
    row.querySelector('.ctx').textContent = data.context_docs ?? '–';
    row.querySelector('.gold').textContent = data.recall == null || !state.goldCount ? '–' : `${Math.round(data.recall * state.goldCount)}/${state.goldCount}`;
    row.querySelector('.secs').textContent = data.seconds == null ? '–' : `${Number(data.seconds).toFixed(1)} s`;
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
    (state.seen[columnName] ||= {})[String(doc.doc_id)] = doc;
    if (doc.gold) markGold(doc.doc_id, columnName);
  }
  return list;
}

function stepCard(type, data, columnName, t) {
  const card = el('details', `event ${type}`);
  card.open = type !== 'retrieve';
  const summary = el('summary');
  const body = el('div', 'body');
  const step = data.step ? ` · bước ${data.step}` : '';
  if (type === 'retrieve') {
    const docs = data.docs || [];
    const hits = docs.filter((d) => d.gold).length;
    summary.textContent = `Truy xuất${step} · ${docs.length} tài liệu`;
    if (hits) summary.append(el('span', 'badge gold-hit', `★ ${hits} gold`));
    if (data.query) body.append(el('div', 'query', `Query: ${data.query}`));
    body.append(docList(data.docs || [], columnName));
  } else if (type === 'thought') {
    summary.textContent = `Suy nghĩ${step}`;
    body.append(el('p', null, data.text));
  } else if (type === 'rd') {
    summary.textContent = `RD${step}: ${data.retrieve ? 'cần truy xuất' : 'không truy xuất'}`;
  } else if (type === 'qf') {
    summary.textContent = `QF${step}`;
    if (data.query) body.append(el('div', 'query', `Query: ${data.query}`));
    if (data.chain && data.chain.length) body.append(el('div', 'muted', `Chuỗi: ${data.chain.map(triple).join(' → ')}`));
  } else if (type === 'ki') {
    summary.textContent = `KI${step}`;
    if (!data.candidates || !data.candidates.length) {
      body.append(el('div', 'muted', 'Không có triple ứng viên (tài liệu chưa có trong cache triple)'));
    } else {
      const picked = Number.isInteger(data.chosen) ? data.candidates[data.chosen] : null;
      if (!picked) {
        body.append(el('div', 'muted', 'Không chọn được triple'));
      } else {
        body.append(el('div', 'triple', triple(picked)));
        const added = data.added_doc ? (state.seen[columnName] || {})[String(data.added_doc)] : null;
        const name = added ? `${added.gold ? '★ ' : ''}${added.title || added.doc_id}` : data.added_doc;
        body.append(el('div', added && added.gold ? 'muted gold-add' : 'muted', data.added_doc ? `Thêm vào ngữ cảnh: ${name}` : 'Tài liệu đã có trong ngữ cảnh'));
      }
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
  const [vText, vCls] = verdict(d);
  if (d.f1 != null) metrics.append(el('span', `verdict-badge ${vCls}`, vText));
  if (d.f1 != null) {
    add(`F1 ${num(d.f1)}`, d.f1 >= 1 ? true : d.f1 <= 0 ? false : null);
    add(`EM ${num(d.em)}`, d.em >= 1);
  }
  if (d.recall != null) add(`Recall ${num(d.recall)}`, d.recall >= 1 ? true : d.recall <= 0 ? false : null);
  add(d.seconds == null ? '–' : `${Number(d.seconds).toFixed(1)} s`);
  add(d.retrievals == null ? '–' : `${d.retrievals} lần truy xuất`);
}

function applyEvent(event) {
  const host = column(event.column);
  if (!host) return;
  const data = event.data || {};
  updateSummary(event.column, event.type, data);
  if (event.type === 'done') { renderMetrics(host, data); return; }
  host.querySelector('.trace').append(stepCard(event.type, data, event.column, event.t || 0));
  if (event.type === 'answer') host.querySelector('footer.answer').textContent = data.text || '(trống)';
  if (event.type === 'error') {
    host.classList.add('failed');
    host.querySelector('.metrics').textContent = `Lỗi: ${data.message || 'không rõ nguyên nhân'}`;
  }
}

function showQuestion(question) {
  const scored = question && question.gold_answers && question.gold_answers.length;
  $('#question-info').hidden = !scored;
  $('#free-note').hidden = !!scored;
  if (!scored) { state.goldCount = 0; return; }
  $('#question-text').textContent = question.question;
  $('#gold-answer').textContent = question.gold_answers.join(' / ');
  for (const d of question.gold_docs || []) if (d.title) state.titles[String(d.id)] = d.title;
  const ids = question.gold_doc_ids || [];
  state.goldCount = ids.length;
  $('#gold-docs').replaceChildren(...ids.map((id) => {
    const item = el('li', null);
    item.dataset.doc = String(id);
    item.title = String(id);
    item.append(el('span', 'gold-id', `★ ${state.titles[String(id)] || id}`), el('span', 'by muted', ''));
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
  if (state.selected && !state.questions.some((q) => q.question === state.selected.question)) {
    state.selected = null;
    $('#eval-search').value = '';
    showQuestion(null);
  }
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
  if (state.stream) { state.stream.close(); state.stream = null; }
  const generation = ++state.generation;
  renderBoard(columns);
  const { run_id } = await api('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (generation !== state.generation) return;
  const stream = new EventSource(`/api/stream/${run_id}`);
  state.stream = stream;
  stream.onmessage = (e) => { try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error('applyEvent failed', err, e.data); notice('Bỏ qua một sự kiện không hiển thị được (xem console).'); } };
  stream.addEventListener('close', () => { stream.onerror = null; stream.close(); });
  stream.onerror = () => { notice('Mất kết nối với server. Lần chạy vẫn được lưu, xem lại trong Lịch sử.'); stream.close(); };
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
  const generation = ++state.generation;
  const events = await api(`/api/replay/${encodeURIComponent(row.file)}`);
  if (generation !== state.generation) return;
  $('#history-dialog').close();
  if (state.stream) { state.stream.close(); state.stream = null; }
  const ex = row.request.example || null;
  const known = ex && state.questions.find((q) => q.id === ex.id);
  showQuestion(known || ex);
  renderBoard([...new Set(events.map((x) => x.column))]);
  const speed = Number($('#replay-speed').value);
  let previous = 0;
  for (const event of events) {
    const t = Number(event.t) || 0;
    await new Promise((resolve) => setTimeout(resolve, Math.max(0, ((t - previous) * 1000) / speed)));
    if (generation !== state.generation) return;
    try { applyEvent(event); } catch (err) { console.error('applyEvent failed', err, event); notice('Bỏ qua một sự kiện không hiển thị được (xem console).'); }
    previous = t;
  }
}

async function refreshHistory() {
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
}
async function showHistory() {
  await refreshHistory();
  if (!$('#history-dialog').open) $('#history-dialog').showModal();
}

async function prerun() {
  try { await api('/api/prerun', { method: 'POST' }); } catch (e) { notice(e.message); return; }
  const poll = async () => {
    let s;
    try { s = await api('/api/prerun'); } catch (e) { notice(e.message); return; }
    $('#prerun-status').textContent = s.error ? `Lỗi: ${s.error}` : s.running ? `Đang chạy ${s.done}/${s.total}…` : `Xong ${s.done}/${s.total}`;
    if (s.running) { setTimeout(poll, 3000); return; }
    if ($('#history-dialog').open) refreshHistory().catch((e) => notice(e.message));
    else showHistory().catch((e) => notice(e.message));
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
$('#focus').onclick = () => {
  const on = document.body.classList.toggle('focus');
  $('#focus').setAttribute('aria-pressed', String(on));
};
$('#theme').onclick = () => {
  const current = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  setTheme(current === 'dark' ? 'light' : 'dark');
};
try { const saved = localStorage.getItem('grasp-theme'); if (saved) document.documentElement.dataset.theme = saved; } catch (e) { /* storage unavailable */ }
loadQuestions().catch((e) => notice(e.message));
health().catch((e) => { $('#health').textContent = `Không kiểm tra được dịch vụ: ${e.message}`; });
