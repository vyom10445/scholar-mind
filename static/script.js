const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('fileInput');
const uploadProgress= document.getElementById('uploadProgress');
const progressFill  = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const chunkCount    = document.getElementById('chunkCount');
const libraryList   = document.getElementById('libraryList');
const emptyState    = document.getElementById('emptyState');
const chatLog        = document.getElementById('chatLog');
const chatForm       = document.getElementById('chatForm');
const chatInput      = document.getElementById('chatInput');
const sendBtn        = document.getElementById('sendBtn');
const settingsBtn    = document.getElementById('settingsBtn');
const settingsPanel  = document.getElementById('settingsPanel');
const closeSettings  = document.getElementById('closeSettings');
const resetBtn       = document.getElementById('resetBtn');

const modelInput  = document.getElementById('modelInput');
const kInput      = document.getElementById('kInput');
const fetchKInput = document.getElementById('fetchKInput');
const lambdaInput = document.getElementById('lambdaInput');
const kVal        = document.getElementById('kVal');
const fetchKVal   = document.getElementById('fetchKVal');
const lambdaVal   = document.getElementById('lambdaVal');

// ---------------------------------------------------------------- library

async function refreshLibrary() {
  const res = await fetch('/api/library');
  const data = await res.json();
  chunkCount.textContent = `${data.total_chunks} chunks`;

  if (data.files.length === 0) {
    libraryList.innerHTML = '<p class="library-empty">No documents yet.</p>';
  } else {
    libraryList.innerHTML = data.files.map(f =>
      `<div class="library-item"><span class="name" title="${f.name}">${f.name}</span><span class="chunks">${f.chunks}</span></div>`
    ).join('');
  }

  emptyState.style.display = data.has_documents ? 'none' : 'flex';
  chatLog.style.display = data.has_documents ? 'flex' : 'none';
}

// ---------------------------------------------------------------- upload

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
});

async function uploadFiles(fileList) {
  const formData = new FormData();
  for (const file of fileList) formData.append('files', file);

  uploadProgress.classList.remove('hidden');
  progressFill.style.width = '15%';
  progressLabel.textContent = `Indexing ${fileList.length} file(s)…`;

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    progressFill.style.width = '100%';
    progressLabel.textContent = `Added ${data.total_chunks_added} chunks`;
    await refreshLibrary();
  } catch (err) {
    progressLabel.textContent = 'Upload failed';
  } finally {
    setTimeout(() => {
      uploadProgress.classList.add('hidden');
      progressFill.style.width = '0%';
    }, 1200);
    fileInput.value = '';
  }
}

// ---------------------------------------------------------------- settings

settingsBtn.addEventListener('click', () => settingsPanel.classList.remove('hidden'));
closeSettings.addEventListener('click', () => settingsPanel.classList.add('hidden'));
settingsPanel.addEventListener('click', e => { if (e.target === settingsPanel) settingsPanel.classList.add('hidden'); });

kInput.addEventListener('input', () => kVal.textContent = kInput.value);
fetchKInput.addEventListener('input', () => fetchKVal.textContent = fetchKInput.value);
lambdaInput.addEventListener('input', () => lambdaVal.textContent = lambdaInput.value);

// ---------------------------------------------------------------- reset

resetBtn.addEventListener('click', async () => {
  if (!confirm('Delete every indexed document? This cannot be undone.')) return;
  await fetch('/api/reset', { method: 'POST' });
  chatLog.innerHTML = '';
  await refreshLibrary();
});

// ---------------------------------------------------------------- chat

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
});
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

function addMessage(role, html) {
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  msg.innerHTML = `
    <div class="avatar ${role}">${role === 'user' ? '🧑‍🎓' : '<img src="/static/assets/logo.png" alt="">'}</div>
    <div class="bubble">${html}</div>
  `;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msg.querySelector('.bubble');
}

function renderSourceChips(sources) {
  if (!sources || !sources.length) return '';
  const chips = sources.map((s, i) => `
    <span class="chip">[${i + 1}] ${s.file} p.${s.page}
      <span class="tip"><b>${s.file}</b> — page ${s.page}<br>${s.snippet}…</span>
    </span>
  `).join('');
  return `<div class="sources">${chips}</div>`;
}

chatForm.addEventListener('submit', async e => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  emptyState.style.display = 'none';
  chatLog.style.display = 'flex';

  addMessage('user', escapeHtml(query));
  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;

  const bubble = addMessage('assistant', '<div class="thinking"><span></span><span></span><span></span></div>');

  const body = {
    query,
    model: modelInput.value,
    k: Number(kInput.value),
    fetch_k: Number(fetchKInput.value),
    lambda_mult: Number(lambdaInput.value),
  };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answer = '';
    let sourcesHtml = '';
    let first = true;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.type === 'sources') {
          sourcesHtml = renderSourceChips(evt.data);
        } else if (evt.type === 'token') {
          if (first) { answer = ''; first = false; }
          answer += evt.data;
          bubble.innerHTML = escapeHtml(answer).replace(/\n/g, '<br>') + sourcesHtml;
          chatLog.scrollTop = chatLog.scrollHeight;
        } else if (evt.type === 'error') {
          bubble.innerHTML = `<span style="color:#F2665B">${escapeHtml(evt.data)}</span>`;
        }
      }
    }
    bubble.innerHTML = escapeHtml(answer).replace(/\n/g, '<br>') + sourcesHtml;
  } catch (err) {
    bubble.innerHTML = '<span style="color:#F2665B">Something went wrong reaching the server.</span>';
  } finally {
    sendBtn.disabled = false;
    chatLog.scrollTop = chatLog.scrollHeight;
  }
});

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------- init

refreshLibrary();