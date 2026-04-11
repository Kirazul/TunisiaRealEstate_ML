const tabs = Array.from(document.querySelectorAll('.book-tab'));
const editors = [];
const charts = [];
const chapterCards = Array.from(document.querySelectorAll('#code-tab-view .chapter-card'));
const workflowStages = Array.from(document.querySelectorAll('.workflow-stage'));
const sourceEditors = [];
const executedCellIndexes = new Set();
let dashboardLoaded = false;

const CELL_DEPENDENCIES = {
    18: [17],
    19: [18],
    20: [18],
    24: [21, 22, 23],
    25: [24],
    26: [24],
    27: [24],
    28: [27],
    29: [27, 28],
    30: [27, 28, 29],
    31: [30],
    32: [31],
    33: [31],
    34: [31],
    38: [35],
    39: [35],
    40: [39],
    41: [40],
    42: [41],
    43: [42],
    44: [43],
    45: [44],
    46: [45],
    47: [46],
    48: [46, 47],
    49: [47, 48],
    50: [49],
    51: [49, 50],
    52: [49, 50, 51],
    53: [52],
    54: [49, 50, 51],
    55: [52],
};

const RUN_SEQUENCE = [];

function tuneEditorLayout(editor) {
    editor.setOption('lineWrapping', true);
    editor.setSize('100%', 'auto');
    const wrapper = editor.getWrapperElement();
    const scroller = editor.getScrollerElement();
    if (wrapper) {
        wrapper.style.width = '100%';
        wrapper.style.height = 'auto';
        wrapper.style.overflow = 'hidden';
    }
    if (scroller) {
        scroller.style.overflow = 'hidden';
        scroller.style.height = 'auto';
        scroller.style.maxHeight = 'none';
    }
    requestAnimationFrame(() => editor.refresh());
}

tabs.forEach((button) => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.book-tab').forEach((node) => node.classList.remove('active'));
        document.querySelectorAll('.tab-view').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
        const tabView = document.getElementById(button.dataset.target);
        tabView.classList.add('active');
        if (button.dataset.target === 'dashboard-tab-view') {
            loadDashboard();
        }
        requestAnimationFrame(() => {
            editors.forEach((editor) => tuneEditorLayout(editor));
            sourceEditors.forEach((editor) => tuneEditorLayout(editor));
            charts.forEach((chart) => {
                if (chart && typeof chart.resize === 'function') chart.resize();
            });
            syncCurrentChapterFromViewport(tabView);
        });
    });
});

document.querySelectorAll('.chapter-nav a').forEach((link) => {
    link.addEventListener('click', () => {
        const tabView = link.closest('.tab-view') || document.querySelector('.tab-view.active');
        if (!tabView) return;
        tabView.querySelectorAll('.chapter-nav a').forEach((node) => node.classList.remove('current-chapter'));
        link.classList.add('current-chapter');
    });
});

async function loadJson(url) {
    const response = await fetch(url);
    return response.json();
}

async function loadProjectJson(path) {
    const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }
    const payload = await response.json();
    return JSON.parse(payload.content);
}

function buildChart(canvasId, labels, data, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    // Chart is loaded via CDN in notebook.html
    // @ts-ignore - Chart is loaded globally
    const chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ label, data, backgroundColor: color, borderRadius: 8 }],
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#e2e8f0' } } },
            scales: {
                x: { ticks: { color: '#e2e8f0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#e2e8f0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            },
        },
    });
    charts.push(chart);
    return chart;
}

function buildDashboardChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (canvas.__chartInstance) {
        canvas.__chartInstance.destroy();
    }
    // @ts-ignore - Chart is loaded globally
    const chart = new Chart(canvas, config);
    canvas.__chartInstance = chart;
    charts.push(chart);
    return chart;
}

function setNodeText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
}

function formatCompactNumber(value) {
    return Number(value || 0).toLocaleString();
}

function formatPercent(value, digits = 2) {
    return `${Number(value || 0).toFixed(digits)}%`;
}

function formatPrice(value) {
    return `${Math.round(Number(value || 0)).toLocaleString()} TND`;
}

function buildDashboardList(containerId, items, mapItem) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    items.forEach((item) => {
        const node = document.createElement('article');
        node.className = 'dashboard-list-item';
        node.innerHTML = mapItem(item);
        container.appendChild(node);
    });
}

function buildFeatureChips(featureColumns = []) {
    const container = document.getElementById('dashboard-feature-chips');
    if (!container) return;
    container.innerHTML = '';
    featureColumns.forEach((feature, index) => {
        const chip = document.createElement('span');
        chip.className = `dashboard-chip${index < 4 ? ' primary' : ''}`;
        chip.textContent = feature;
        container.appendChild(chip);
    });
}

function setSceneStatus(text) {
    const node = document.getElementById('scene-status');
    if (node) node.textContent = text;
}

function updateCurrentChapter(tabView, chapterId) {
    if (!tabView) return;
    tabView.querySelectorAll('.chapter-nav a').forEach((link) => {
        const href = link.getAttribute('href') || '';
        link.classList.toggle('current-chapter', href === `#${chapterId}`);
    });

    if (tabView.id === 'code-tab-view') {
        workflowStages.forEach((stage) => {
            stage.classList.toggle('active', stage.dataset.stage === chapterId);
            stage.classList.remove('completed');
        });
    }
}

function syncCurrentChapterFromViewport(tabView = document.querySelector('.tab-view.active')) {
    if (!tabView) return;
    const cards = Array.from(tabView.querySelectorAll('.chapter-card'));
    if (!cards.length) return;

    const viewportAnchor = window.innerHeight * 0.28;
    let bestCard = cards[0];
    let bestDistance = Number.POSITIVE_INFINITY;

    cards.forEach((card) => {
        const rect = card.getBoundingClientRect();
        const distance = Math.abs(rect.top - viewportAnchor);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestCard = card;
        }
    });

    updateCurrentChapter(tabView, bestCard.id);
}

let scrollSyncFrame = null;

function requestScrollSync() {
    if (scrollSyncFrame !== null) return;
    scrollSyncFrame = requestAnimationFrame(() => {
        scrollSyncFrame = null;
        syncCurrentChapterFromViewport();
    });
}

function prettifyStdout(stdout) {
    const wrapper = document.createElement('div');
    wrapper.className = 'output-grid';
    const text = String(stdout || '').trim();
    if (!text) return wrapper;

    const blocks = splitOutputIntoBlocks(text);
    blocks.forEach((block) => {
        const rendered = renderOutputBlock(block.join('\n'));
        if (rendered) wrapper.appendChild(rendered);
    });

    return wrapper;
}

function splitOutputIntoBlocks(text) {
    const lines = text.split(/\r?\n/);
    const blocks = [];
    let current = [];

    function flush() {
        if (current.length) {
            blocks.push(current);
            current = [];
        }
    }

    function isKvLine(line) {
        return /^([^:=]+?)\s*[:=]\s*(.+)$/.test(line.trim());
    }

    function isListLine(line) {
        const trimmed = line.trim();
        return trimmed.startsWith('[') && trimmed.endsWith(']');
    }

    function isTableLine(line) {
        const trimmed = line.trim();
        if (!trimmed) return false;
        return /\s{2,}/.test(trimmed) || /^\d+\s{2,}/.test(trimmed);
    }

    function lineType(line) {
        if (!line.trim()) return 'blank';
        if (isListLine(line)) return 'list';
        if (isKvLine(line)) return 'kv';
        if (isTableLine(line)) return 'table';
        return 'text';
    }

    let lastType = null;
    for (const line of lines) {
        const type = lineType(line);
        if (type === 'blank') {
            flush();
            lastType = null;
            continue;
        }
        if (lastType && type !== lastType && !(lastType === 'table' && type === 'table')) {
            flush();
        }
        current.push(line);
        lastType = type;
    }
    flush();
    return blocks;
}

function renderOutputBlock(block) {
    const lines = block.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (!lines.length) return null;

    const kvLines = [];
    const otherLines = [];
    lines.forEach((line) => {
        const match = line.match(/^([^:=]+?)\s*[:=]\s*(.+)$/);
        if (match && match[1].trim().length < 60) {
            kvLines.push([match[1].trim(), match[2].trim()]);
        } else {
            otherLines.push(line);
        }
    });

    const listCard = tryRenderListBlock(block);
    if (listCard) return listCard;

    const tableCard = tryRenderTableBlock(lines);
    if (tableCard) return tableCard;

    if (kvLines.length && kvLines.length >= Math.max(1, otherLines.length)) {
        const section = document.createElement('section');
        section.className = 'output-section output-section-metrics';
        const metrics = document.createElement('div');
        metrics.className = 'output-metrics';
        kvLines.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'output-metric-card';
            row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
            metrics.appendChild(row);
        });
        section.appendChild(metrics);
        if (otherLines.length) {
            const pre = document.createElement('div');
            pre.className = 'output-pre';
            pre.textContent = otherLines.join('\n');
            section.appendChild(pre);
        }
        return section;
    }

    const pre = document.createElement('div');
    pre.className = 'output-pre output-section';
    pre.textContent = block;
    return pre;
}

function tryRenderListBlock(block) {
    const trimmed = block.trim();
    if (!(trimmed.startsWith('[') && trimmed.endsWith(']'))) return null;
    const items = Array.from(trimmed.matchAll(/'([^']+)'|"([^"]+)"/g)).map((match) => match[1] || match[2]).filter(Boolean);
    if (!items.length) return null;

    const section = document.createElement('section');
    section.className = 'output-section output-section-list';
    section.innerHTML = `<div class="output-section-head"><span>Detected Items</span><strong>${items.length}</strong></div>`;
    const cloud = document.createElement('div');
    cloud.className = 'output-chip-cloud';
    items.forEach((item) => {
        const chip = document.createElement('span');
        chip.className = 'output-chip';
        chip.textContent = item;
        cloud.appendChild(chip);
    });
    section.appendChild(cloud);
    return section;
}

function splitTableLine(line) {
    return line.trim().split(/\s{2,}/).map((part) => part.trim()).filter(Boolean);
}

function tryRenderTableBlock(lines) {
    if (lines.length < 2) return null;
    if (!lines.some((line) => /\s{2,}/.test(line.trim()))) return null;
    const headerParts = splitTableLine(lines[0]);
    const rowParts = splitTableLine(lines[1]);
    if (headerParts.length < 2 || rowParts.length < 2) return null;

    const normalizedRows = lines.map(splitTableLine).filter((parts) => parts.length >= 2);
    if (normalizedRows.length < 2) return null;

    let columns = headerParts;
    let dataRows = normalizedRows.slice(1);
    if (rowParts.length === headerParts.length + 1 || /^\d+$/.test(rowParts[0])) {
        columns = ['#', ...headerParts];
        dataRows = normalizedRows.slice(1).map((row) => row.length === columns.length ? row : [''].concat(row));
    }
    const validRows = dataRows.filter((row) => row.length === columns.length).slice(0, 12);
    if (!validRows.length) return null;

    const section = document.createElement('section');
    section.className = 'output-section output-section-table';
    section.innerHTML = `<div class="output-section-head"><span>Structured Output</span><strong>${validRows.length} rows</strong></div>`;

    const tableWrap = document.createElement('div');
    tableWrap.className = 'output-table-wrap';
    const table = document.createElement('table');
    table.className = 'output-table';

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach((column) => {
        const th = document.createElement('th');
        th.textContent = column;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    validRows.forEach((row) => {
        const tr = document.createElement('tr');
        row.forEach((value) => {
            const td = document.createElement('td');
            td.textContent = value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    section.appendChild(tableWrap);
    return section;
}

function getModeFromPath(path) {
    if (path.endsWith('.html')) return 'htmlmixed';
    if (path.endsWith('.css')) return 'css';
    if (path.endsWith('.js')) return 'javascript';
    if (path.endsWith('.json')) return 'javascript';
    return 'python';
}

async function loadStoryAndDashboard() {
    try {
        const modelSummary = await loadJson('/model_summary');
        const trainingRows = Number(modelSummary.training_rows || 0);
        const directCoverage = Number(modelSummary.direct_coverage_pct || 0);
        const validationR2 = Number(modelSummary.accuracy_pct || 0);

        const heroTraining = document.getElementById('hero-training');
        const heroDirect = document.getElementById('hero-direct');
        const heroModel = document.getElementById('hero-model');
        const heroScore = document.getElementById('hero-score');

        if (heroTraining) heroTraining.textContent = trainingRows.toLocaleString();
        if (heroDirect) heroDirect.textContent = `${directCoverage.toFixed(2)}%`;
        if (heroModel) heroModel.textContent = modelSummary.best_model || 'N/A';
        if (heroScore) heroScore.textContent = `${validationR2.toFixed(2)}%`;

        const cvContainer = document.getElementById('hero-cv');
        if (cvContainer && Number.isFinite(Number(modelSummary.cv_r2_mean))) {
            cvContainer.textContent = `${(Number(modelSummary.cv_r2_mean) * 100).toFixed(2)}% ± ${(Number(modelSummary.cv_r2_std || 0) * 100).toFixed(2)}%`;
        }
    } catch (error) {
        console.error('Failed to load notebook summary', error);
    }
}

async function loadDashboard() {
    if (dashboardLoaded) return;
    dashboardLoaded = true;

    try {
        const [
            discovery,
            cleaning,
            merge,
            geo,
            training,
            features,
            model,
        ] = await Promise.all([
            loadProjectJson('data/processed/01_discovery/01_merge_overview.json'),
            loadProjectJson('data/processed/02_cleaning/02_merge_readiness.json'),
            loadProjectJson('data/processed/03_merge/03_merge_report.json'),
            loadProjectJson('data/processed/04_geo_alignment/04_geo_alignment_report.json'),
            loadProjectJson('data/processed/05_training_dataset/05_training_dataset_report.json'),
            loadProjectJson('data/processed/06_feature_engineering/06_feature_engineering_report.json'),
            loadJson('/model_summary'),
        ]);

        const rawRows = (discovery.datasets || []).reduce((sum, dataset) => sum + Number(dataset.rows || 0), 0);
        const cleanRows = (cleaning.datasets || []).reduce((sum, dataset) => sum + Number(dataset.rows || 0), 0);
        const geoMatched = Number(geo.matched_rows || 0);
        const modelingRows = Number(model.modeling_rows || 0);
        const retention = rawRows ? (modelingRows / rawRows) * 100 : 0;

        setNodeText('dash-raw-rows', formatCompactNumber(rawRows));
        setNodeText('dash-clean-rows', formatCompactNumber(cleanRows));
        setNodeText('dash-geo-matched', formatCompactNumber(geoMatched));
        setNodeText('dash-modeling-rows', formatCompactNumber(modelingRows));
        setNodeText('dash-retention-rate', formatPercent(retention));
        setNodeText('dash-atlas-reach', formatPercent(model.atlas_reach_pct || 0));
        setNodeText('dash-validation-score', formatPercent(model.accuracy_pct || 0));

        setNodeText('dash-canonical-delegations', formatCompactNumber(geo.geo_canonical_delegations || 0));
        setNodeText('dash-covered-delegations', formatCompactNumber(geo.unique_geo_delegations_covered || 0));
        setNodeText('dash-direct-delegations', formatCompactNumber(model.delegations_with_direct_support || 0));
        setNodeText('dash-unmatched-rows', formatCompactNumber(geo.unmatched_rows || 0));

        setNodeText('dash-price-min', formatPrice(training.price_range?.min || 0));
        setNodeText('dash-price-median', formatPrice(training.price_range?.median || 0));
        setNodeText('dash-price-max', formatPrice(training.price_range?.max || 0));

        buildDashboardChart('dashboard-discovery-chart', {
            type: 'bar',
            data: {
                labels: (discovery.datasets || []).map((item) => item.dataset_name.replaceAll('_', ' ')),
                datasets: [{
                    label: 'Raw rows',
                    data: (discovery.datasets || []).map((item) => Number(item.rows || 0)),
                    backgroundColor: ['#7f1d1d', '#b91c1c', '#ef4444'],
                    borderRadius: 10,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#e2e8f0' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                },
            },
        });

        buildDashboardChart('dashboard-cleaning-chart', {
            type: 'bar',
            data: {
                labels: (cleaning.datasets || []).map((item) => item.dataset_name.replaceAll('_', ' ')),
                datasets: [{
                    label: 'Clean rows',
                    data: (cleaning.datasets || []).map((item) => Number(item.rows || 0)),
                    backgroundColor: ['rgba(248,113,113,0.92)', 'rgba(251,146,60,0.92)', 'rgba(244,63,94,0.92)'],
                    borderRadius: 999,
                }],
            },
            options: {
                indexAxis: 'y',
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#e2e8f0' }, grid: { display: false } },
                },
            },
        });

        buildDashboardChart('dashboard-merge-chart', {
            type: 'doughnut',
            data: {
                labels: Object.keys(merge.source_distribution || {}).map((item) => item.replaceAll('_', ' ')),
                datasets: [{
                    data: Object.values(merge.source_distribution || {}).map((value) => Number(value || 0)),
                    backgroundColor: ['#991b1b', '#dc2626', '#f87171'],
                    borderColor: '#0f172a',
                    borderWidth: 3,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#e2e8f0', boxWidth: 12 } } },
                cutout: '68%',
            },
        });

        buildDashboardChart('dashboard-geo-match-chart', {
            type: 'doughnut',
            data: {
                labels: Object.keys(geo.match_status || {}).map((item) => item.replaceAll('_', ' ')),
                datasets: [{
                    data: Object.values(geo.match_status || {}).map((value) => Number(value || 0)),
                    backgroundColor: ['#ef4444', '#fb7185', '#f97316', '#f59e0b', '#7f1d1d', '#334155'],
                    borderColor: '#0f172a',
                    borderWidth: 3,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#e2e8f0', boxWidth: 12 } } },
                cutout: '62%',
            },
        });

        const governorateEntries = Object.entries(geo.geo_counts_by_governorate || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
        buildDashboardChart('dashboard-governorate-chart', {
            type: 'bar',
            data: {
                labels: governorateEntries.map(([name]) => name),
                datasets: [{
                    label: 'Delegations',
                    data: governorateEntries.map(([, count]) => Number(count || 0)),
                    backgroundColor: 'rgba(248,113,113,0.86)',
                    borderRadius: 8,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#cbd5e1', maxRotation: 55, minRotation: 55 }, grid: { display: false } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                },
            },
        });

        buildDashboardChart('dashboard-coverage-chart', {
            type: 'pie',
            data: {
                labels: ['Direct support', 'Fallback coverage'],
                datasets: [{
                    data: [Number(model.direct_coverage_pct || 0), Number(model.fallback_support_pct || 0)],
                    backgroundColor: ['#ef4444', '#7f1d1d'],
                    borderColor: '#0f172a',
                    borderWidth: 3,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#e2e8f0', boxWidth: 12 } } },
            },
        });

        buildDashboardChart('dashboard-family-chart', {
            type: 'polarArea',
            data: {
                labels: Object.keys(training.family_distribution || {}).map((item) => item.replaceAll('_', ' ')),
                datasets: [{
                    data: Object.values(training.family_distribution || {}).map((value) => Number(value || 0)),
                    backgroundColor: ['rgba(239,68,68,0.88)', 'rgba(251,146,60,0.82)', 'rgba(248,113,113,0.68)'],
                    borderColor: '#111827',
                    borderWidth: 2,
                }],
            },
            options: {
                maintainAspectRatio: false,
                scales: { r: { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8', backdropColor: 'transparent' } } },
                plugins: { legend: { position: 'bottom', labels: { color: '#e2e8f0', boxWidth: 12 } } },
            },
        });

        buildDashboardChart('dashboard-model-chart', {
            type: 'bar',
            data: {
                labels: (model.model_results || []).map((item) => item.model),
                datasets: [{
                    label: 'Validation R²',
                    data: (model.model_results || []).map((item) => Number(item.r2 || 0)),
                    backgroundColor: ['#ef4444', '#475569'],
                    borderRadius: 10,
                }],
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#e2e8f0' }, grid: { display: false } },
                    y: { min: 0, max: 1, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                },
            },
        });

        buildDashboardList('dashboard-unmatched-list', (geo.top_unmatched_pairs || []).slice(0, 5), (item) => `
            <span>${item.governorate}</span>
            <strong>${item.city}</strong>
            <p>${Number(item.count || 0).toLocaleString()} unresolved row(s)</p>
        `);

        buildFeatureChips(model.feature_columns || features.feature_columns || []);
        setSceneStatus('Dashboard analytics loaded');
    } catch (error) {
        console.error('Failed to load dashboard analytics', error);
        dashboardLoaded = false;
        setSceneStatus('Dashboard unavailable');
    }
}

async function loadSources() {
    const panels = Array.from(document.querySelectorAll('.source-panel'));
    await Promise.all(panels.map(async (panel) => {
        const path = panel.dataset.sourcePath;
        const codeContainer = panel.querySelector('.source-code');
        const header = panel.querySelector('.source-header');
        const actions = document.createElement('div');
        actions.className = 'source-actions';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.type = 'button';
        copyBtn.textContent = 'COPY';
        actions.appendChild(copyBtn);
        if (header && !header.querySelector('.source-actions')) header.appendChild(actions);

        try {
            const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`);
            const payload = await response.json();
            codeContainer.innerHTML = '';
            const viewerHost = document.createElement('div');
            viewerHost.className = 'source-viewer';
            codeContainer.appendChild(viewerHost);
            const editor = CodeMirror(viewerHost, {
                value: payload.content,
                mode: getModeFromPath(path),
                theme: 'dracula',
                lineNumbers: true,
                lineWrapping: true,
                readOnly: true,
                viewportMargin: Infinity,
            });
            sourceEditors.push(editor);
            tuneEditorLayout(editor);
        copyBtn.addEventListener('click', async () => {
            await navigator.clipboard.writeText(payload.content);
            copyBtn.textContent = 'COPIED';
            setTimeout(() => { copyBtn.textContent = 'COPY'; }, 1200);
        });
    } catch (error) {
        codeContainer.textContent = 'Failed to load source file.';
    }
    }));
}

async function executeCell(cell, editor, index) {
    const runBtn = cell.querySelector('.btn-run');
    const output = cell.querySelector('.output');
    const label = cell.querySelector('.cell-label');
    const chapter = cell.closest('.chapter-card');
    if (chapter) {
        setSceneStatus(`Running: ${chapter.querySelector('h2').textContent}`);
    }
    label.textContent = 'In [*]:';
    runBtn.disabled = true;
    output.classList.add('pretty-output');
    output.innerHTML = '<div class="output-pre">Executing...</div>';
    try {
        const response = await fetch('/run-cell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: editor.getValue() }),
        });
        if (!response.ok) {
            throw new Error(`Run cell failed with ${response.status}`);
        }
        const data = await response.json();
        label.textContent = `In [${index + 1}]:`;
        output.innerHTML = '';
        output.appendChild(prettifyStdout(data.stdout || ''));
        if (data.stderr) {
            const err = document.createElement('div');
            err.className = 'output-error';
            err.textContent = data.stderr;
            output.appendChild(err);
        }
        if (data.plot) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.plot}`;
            output.appendChild(img);
        }
        if (!data.stderr) {
            executedCellIndexes.add(index);
        }
    } catch (error) {
        output.innerHTML = '<div class="output-error">Server connection failed.</div>';
    } finally {
        runBtn.disabled = false;
    }
}

async function ensureCellReady(index, visiting = new Set()) {
    if (executedCellIndexes.has(index)) return;
    if (visiting.has(index)) return;
    visiting.add(index);

    const prereqs = CELL_DEPENDENCIES[index] || [];
    for (const prereqIndex of prereqs) {
        await ensureCellReady(prereqIndex, visiting);
    }

    if (!executedCellIndexes.has(index)) {
        const allCells = Array.from(document.querySelectorAll('.cell'));
        const cell = allCells[index];
        const editor = editors[index];
        if (cell && editor) {
            await executeCell(cell, editor, index);
        }
    }

    visiting.delete(index);
}

function initEditors() {
    document.querySelectorAll('.code-editor').forEach((textarea, index) => {
        const editor = CodeMirror.fromTextArea(textarea, {
            mode: 'python',
            theme: 'dracula',
            lineNumbers: false,
            lineWrapping: true,
            indentUnit: 4,
            viewportMargin: Infinity,
        });
        editors.push(editor);
        tuneEditorLayout(editor);
    });
}

function runCell(button) {
    const cell = button.closest('.cell');
    const allCells = Array.from(document.querySelectorAll('.cell'));
    const index = allCells.indexOf(cell);
    if (!cell || index < 0 || !editors[index]) return;
    return ensureCellReady(index);
}

async function resetKernel() {
    await fetch('/run-cell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: '', reset: true }),
    });
    document.querySelectorAll('.output').forEach((node) => { node.innerHTML = ''; node.classList.remove('pretty-output'); });
    document.querySelectorAll('.cell-label').forEach((node) => { node.textContent = 'In [ ]:'; });
    executedCellIndexes.clear();
    workflowStages.forEach((stage) => stage.classList.remove('completed'));
    setSceneStatus('Kernel reset');
}

async function runAllCells() {
    document.querySelector('[data-target="code-tab-view"]').click();
    await resetKernel();
    const sequence = RUN_SEQUENCE.length
        ? RUN_SEQUENCE.filter((index) => index < editors.length)
        : Array.from({ length: editors.length }, (_, index) => index);
    for (const i of sequence) {
        if (executedCellIndexes.has(i)) continue;
        const cell = document.querySelectorAll('.cell')[i];
        const chapter = cell.closest('.chapter-card');
        if (chapter) {
            setSceneStatus(`Running ${chapter.querySelector('h2').textContent}`);
            await new Promise((resolve) => setTimeout(resolve, 500));
        }
        await ensureCellReady(i);
        if (chapter) {
            const chapterCells = Array.from(chapter.querySelectorAll('.cell'));
            const chapterIndex = chapterCells.indexOf(cell);
            if (chapterIndex === chapterCells.length - 1) {
            setSceneStatus(`Completed ${chapter.querySelector('h2').textContent}`);
            await new Promise((resolve) => setTimeout(resolve, 400));
            }
        }
    }
    setSceneStatus('Full lab completed');
}

function openRawNotebook() {
    window.open('/api/source/rendered?path=RealEstate_Complete_Pipeline.ipynb', '_blank', 'noopener');
}

window.resetKernel = resetKernel;
window.runAllCells = runAllCells;
window.runCell = runCell;
window.openRawNotebook = openRawNotebook;

initEditors();
loadStoryAndDashboard();
loadSources();
setSceneStatus('Ready to run');

const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (visible) {
        const activeTab = document.querySelector('.tab-view.active');
        updateCurrentChapter(activeTab, visible.target.id);
    }
}, { threshold: [0.12, 0.22, 0.38], rootMargin: '-8% 0px -52% 0px' });

document.querySelectorAll('.chapter-card').forEach((card) => observer.observe(card));

syncCurrentChapterFromViewport();
window.addEventListener('scroll', requestScrollSync, { passive: true });
window.addEventListener('resize', requestScrollSync);
