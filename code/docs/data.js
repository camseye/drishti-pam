/* =========================================================
 * Drishti-PAM dashboard — Data Source view
 * ─────────────────────────────────────────────────────────
 * दृष्टि — the gaze that sees beyond the surface.
 * =========================================================
 * Pure ES2020. Loaded with `defer` AFTER app.js so it can
 * piggy-back on `app.js`-owned assets (theme, scenario
 * vocabulary if available) without racing it.
 *
 * Owns the second tab of the SPA — the `#/data` view that
 * shows the raw synthetic corpus the detectors were fed.
 *
 * Lifecycle:
 *   - Listens for `pam:data-view-shown` (dispatched by app.js
 *     whenever the data tab becomes visible) and lazy-loads
 *     `data/sessions_raw.json` on first reveal — so users who
 *     never click the tab don't pay the download cost.
 *   - Re-renders on filter / sort / page changes.
 *
 * The data view is intentionally read-only — clicking a row
 * does NOT open the drawer (that's a triage-side concept).
 * Users wanting to see what the model did with a row are
 * directed back to the triage tab via the page subtitle link.
 * ========================================================= */
(() => {
  'use strict';

  // ----------------------------------------------------------
  //  State (parallel to app.js's `state`, but isolated)
  // ----------------------------------------------------------
  const state = {
    rows: null,            // null until first load attempt
    loading: null,         // in-flight Promise (de-dupe concurrent shows)
    loaded: false,
    filters: {
      split: '',
      label_gt: '',
      scenario_gt: '',
      ticket_category: '',
      search: '',
    },
    sort: { col: 'id', dir: 'asc' },
    page: 1,
    pageSize: 50,
    visible: [],
  };

  // Plain-English scenario labels — kept in sync with the
  // generator's catalogue. If `app.js` exposes its own
  // SCENARIO_INFO via `window.PAM_SCENARIO_INFO` we'll prefer
  // that (single source of truth); otherwise we fall back to
  // a local copy so this file stays self-sufficient.
  const SCENARIO_INFO_FALLBACK = {
    S1:  { short: 'Multi-user ticket' },
    S2:  { short: 'Vague justification' },
    S3:  { short: 'Off-window session' },
    S4:  { short: 'Wrong-category commands' },
    S5:  { short: 'Off-hours work' },
    S7:  { short: 'Destructive on read-only' },
    S10: { short: 'Out-of-scope asset' },
    S11: { short: 'Zombie ticket' },
    S12: { short: 'Sensitive-data probing' },
    S13: { short: 'Lateral movement' },
    S14: { short: 'Privilege escalation' },
    S15: { short: 'Approver-as-implementer' },
    S16: { short: 'Vocabulary-cloak adversarial' },
    clean: { short: 'No flagged scenario' },
  };

  // Length cap for the inline justification preview. Anything
  // beyond gets ellipsised with the full text shown on hover.
  const JUSTIFICATION_PREVIEW_CHARS = 80;

  // Ordering used when `split` is the sort key. train < test < unknown
  // is intentional — train is the most common, test is the
  // evaluation surface, unknown is data-quality noise.
  const SPLIT_ORDER = { train: 0, test: 1, unknown: 2 };
  const LABEL_ORDER = { suspicious: 0, clean: 1 };

  // ----------------------------------------------------------
  //  Boot
  // ----------------------------------------------------------

  // Wire one-shot listener immediately. The first time the user
  // navigates to #/data, app.js fires this event — we lazy-load,
  // render, and then forward subsequent events into a re-render.
  window.addEventListener('pam:data-view-shown', onShown);

  // If the page loaded with `#/data` already in the URL,
  // app.js dispatches the event during its own boot —
  // but if we registered after that ran, we'd miss it.
  // Catch up by checking the hash on DOMContentLoaded.
  document.addEventListener('DOMContentLoaded', () => {
    if ((window.location.hash || '').startsWith('#/data')) {
      onShown();
    }
    wireFilters();
    wireSorting();
    wirePagination();
  });

  async function onShown() {
    if (!state.loaded && !state.loading) {
      state.loading = loadRaw();
    }
    if (state.loading) {
      try {
        await state.loading;
      } catch (err) {
        console.error('[data] load failed:', err);
        renderLoadError(err);
        return;
      }
    }
    populateScenarioFilter();
    populateCategoryFilter();
    recompute();
  }

  async function loadRaw() {
    const t0 = performance.now();
    const res = await fetch('data/sessions_raw.json', {cache: 'no-cache'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows = await res.json();
    state.rows = Array.isArray(rows) ? rows : [];
    state.loaded = true;
    state.loading = null;
    const ms = (performance.now() - t0).toFixed(0);
    // eslint-disable-next-line no-console
    console.info(`[data] loaded ${state.rows.length.toLocaleString()} `
                 + `raw sessions in ${ms} ms`);
    setBind('total_count', fmtInt(state.rows.length));
    const totalEl = document.querySelector('[data-bind="raw_total"]');
    if (totalEl) totalEl.textContent = fmtInt(state.rows.length);
  }

  // ----------------------------------------------------------
  //  Filter + sort + paginate pipeline
  // ----------------------------------------------------------

  function wireFilters() {
    document.querySelectorAll('[data-data-filter]').forEach(el => {
      const key = el.dataset.dataFilter;
      const evt = el.tagName === 'SELECT' ? 'change' : 'input';
      el.addEventListener(evt, () => {
        state.filters[key] = el.value.trim();
        state.page = 1;
        recompute();
      });
    });
    const clearBtn = document.querySelector('[data-action="clear-data-filters"]');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        state.filters = {
          split: '', label_gt: '', scenario_gt: '',
          ticket_category: '', search: '',
        };
        document.querySelectorAll('[data-data-filter]').forEach(el => {
          el.value = '';
        });
        state.page = 1;
        recompute();
      });
    }
  }

  function wireSorting() {
    document.querySelectorAll('th[data-data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.dataSort;
        if (state.sort.col === col) {
          state.sort.dir = state.sort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          state.sort.col = col;
          state.sort.dir = 'asc';
        }
        renderSortIndicators();
        recompute();
      });
    });
    renderSortIndicators();
  }

  function renderSortIndicators() {
    document.querySelectorAll('th[data-data-sort]').forEach(th => {
      th.removeAttribute('data-data-sort-active');
      if (th.dataset.dataSort === state.sort.col) {
        th.setAttribute('data-data-sort-active', state.sort.dir);
      }
    });
  }

  function wirePagination() {
    const prev = document.querySelector('[data-action="data-prev-page"]');
    const next = document.querySelector('[data-action="data-next-page"]');
    if (prev) prev.addEventListener('click', () => {
      if (state.page > 1) { state.page -= 1; recompute(); }
    });
    if (next) next.addEventListener('click', () => {
      const totalPages = Math.max(1, Math.ceil(state.visible.length / state.pageSize));
      if (state.page < totalPages) { state.page += 1; recompute(); }
    });
  }

  function recompute() {
    if (!state.loaded || !Array.isArray(state.rows)) return;
    const filtered = applyFilters(state.rows, state.filters);
    state.visible = sortRows(filtered, state.sort);
    setBind('filtered_count', fmtInt(state.visible.length));
    renderTable();
    renderPagination();
  }

  function applyFilters(rows, f) {
    const search = f.search.toLowerCase();
    return rows.filter(r => {
      if (f.split && r.split !== f.split) return false;
      if (f.label_gt && r.label_gt !== f.label_gt) return false;
      if (f.scenario_gt && r.scenario_gt !== f.scenario_gt) return false;
      if (f.ticket_category && r.ticket_category !== f.ticket_category) return false;
      if (search) {
        const hay = `${r.user} ${r.host} ${r.ticket_id} ${r.ticket_approver}`.toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    });
  }

  function sortRows(rows, sort) {
    const dir = sort.dir === 'desc' ? -1 : 1;
    const out = rows.slice();
    out.sort((a, b) => dir * compareForCol(a, b, sort.col));
    return out;
  }

  function compareForCol(a, b, col) {
    if (col === 'split') {
      return (SPLIT_ORDER[a.split] ?? 99) - (SPLIT_ORDER[b.split] ?? 99);
    }
    if (col === 'label_gt') {
      return (LABEL_ORDER[a.label_gt] ?? 99) - (LABEL_ORDER[b.label_gt] ?? 99);
    }
    const av = a[col];
    const bv = b[col];
    if (typeof av === 'number' && typeof bv === 'number') return av - bv;
    const as = (av ?? '').toString();
    const bs = (bv ?? '').toString();
    return as.localeCompare(bs);
  }

  // ----------------------------------------------------------
  //  Filter dropdown population
  // ----------------------------------------------------------

  function populateScenarioFilter() {
    const sel = document.querySelector('#data-filter-scenario');
    if (!sel || sel.options.length > 1) return; // only populate once
    const seen = new Set();
    state.rows.forEach(r => seen.add(r.scenario_gt));
    const order = ['clean', 'S1', 'S2', 'S3', 'S4', 'S5',
                   'S7', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16'];
    const ordered = order.filter(s => seen.has(s));
    seen.forEach(s => { if (!ordered.includes(s)) ordered.push(s); });

    const info = window.PAM_SCENARIO_INFO || SCENARIO_INFO_FALLBACK;
    ordered.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      const meta = info[s];
      opt.textContent = meta ? `${meta.short} (${s})` : s;
      sel.appendChild(opt);
    });
  }

  function populateCategoryFilter() {
    const sel = document.querySelector('#data-filter-category');
    if (!sel || sel.options.length > 1) return; // only populate once
    const seen = new Set();
    state.rows.forEach(r => seen.add(r.ticket_category));
    const sorted = Array.from(seen).sort();
    sorted.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    });
  }

  // ----------------------------------------------------------
  //  Table rendering
  // ----------------------------------------------------------

  function renderTable() {
    const tbody = document.querySelector('#data-tbody');
    if (!tbody) return;
    tbody.textContent = '';

    if (state.visible.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 15;
      td.className = 'table-empty';
      td.textContent = 'No sessions match the current filters.';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    const start = (state.page - 1) * state.pageSize;
    const end = Math.min(start + state.pageSize, state.visible.length);
    const slice = state.visible.slice(start, end);
    const frag = document.createDocumentFragment();
    slice.forEach(r => frag.appendChild(buildRow(r)));
    tbody.appendChild(frag);
  }

  function renderPagination() {
    const totalPages = Math.max(1, Math.ceil(state.visible.length / state.pageSize));
    setBind('page_current', String(state.page));
    setBind('page_total',   String(totalPages));
    const prev = document.querySelector('[data-action="data-prev-page"]');
    const next = document.querySelector('[data-action="data-next-page"]');
    if (prev) prev.disabled = state.page <= 1;
    if (next) next.disabled = state.page >= totalPages;
  }

  function buildRow(r) {
    const tr = document.createElement('tr');
    tr.appendChild(td(monoText(r.id), 'mono'));
    tr.appendChild(td(splitBadge(r.split)));
    tr.appendChild(td(labelPill(r.label_gt)));
    tr.appendChild(td(scenarioCell(r.scenario_gt)));
    tr.appendChild(td(monoText(r.user), 'mono'));
    tr.appendChild(td(monoText(r.host), 'mono'));
    tr.appendChild(td(monoText(r.ticket_id), 'mono'));
    tr.appendChild(td(text(r.ticket_category)));
    tr.appendChild(td(monoText(r.ticket_approver), 'mono'));
    tr.appendChild(td(timestampSpan(r.start_time), 'mono'));
    tr.appendChild(td(changeWindowSpan(r.change_window_start, r.change_window_end), 'mono'));
    tr.appendChild(td(text(fmtInt(r.command_count)), 'num'));
    tr.appendChild(td(text(fmtInt(r.named_assets_count)), 'num'));
    tr.appendChild(td(text(fmtInt(r.vault_fetches_count)), 'num'));
    tr.appendChild(td(justificationCell(r.justification)));
    return tr;
  }

  function td(child, cls) {
    const cell = document.createElement('td');
    if (cls) cell.className = cls;
    if (typeof child === 'string') cell.textContent = child;
    else if (child) cell.appendChild(child);
    return cell;
  }

  function text(s) {
    return document.createTextNode(s == null ? '—' : String(s));
  }

  function monoText(s) {
    const span = document.createElement('span');
    span.textContent = s == null ? '—' : String(s);
    return span;
  }

  function splitBadge(split) {
    const span = document.createElement('span');
    const norm = split || 'unknown';
    span.className = `split-badge split-badge--${norm}`;
    span.textContent = norm;
    return span;
  }

  function labelPill(label) {
    const span = document.createElement('span');
    const norm = label === 'suspicious' ? 'suspicious' : 'clean';
    span.className = `label-pill label-pill--${norm}`;
    span.textContent = norm;
    return span;
  }

  function scenarioCell(s) {
    const info = window.PAM_SCENARIO_INFO || SCENARIO_INFO_FALLBACK;
    const meta = info[s];
    const span = document.createElement('span');
    if (meta) {
      span.textContent = `${meta.short} (${s})`;
      if (meta.long) span.title = meta.long;
    } else {
      span.textContent = s || '—';
    }
    return span;
  }

  function timestampSpan(iso) {
    if (!iso) return text('—');
    const span = document.createElement('span');
    span.className = 'ts';
    span.textContent = iso.replace('T', ' ');
    return span;
  }

  function changeWindowSpan(startIso, endIso) {
    if (!startIso && !endIso) return text('—');
    const span = document.createElement('span');
    span.className = 'ts';
    const s = startIso ? startIso.slice(5, 16).replace('T', ' ') : '?';
    const e = endIso   ? endIso.slice(5, 16).replace('T', ' ')   : '?';
    span.textContent = `${s} → ${e}`;
    span.title = `${startIso || '?'}  →  ${endIso || '?'}`;
    return span;
  }

  function justificationCell(just) {
    const span = document.createElement('span');
    span.className = 'just-cell';
    if (!just) {
      span.textContent = '—';
      return span;
    }
    if (just.length > JUSTIFICATION_PREVIEW_CHARS) {
      span.textContent = just.slice(0, JUSTIFICATION_PREVIEW_CHARS - 1) + '…';
    } else {
      span.textContent = just;
    }
    span.title = just;
    return span;
  }

  // ----------------------------------------------------------
  //  Tiny utilities
  // ----------------------------------------------------------

  function setBind(key, value) {
    document.querySelectorAll(`[data-bind-data="${key}"]`).forEach(el => {
      el.textContent = value;
    });
  }

  function fmtInt(n) {
    return Number(n || 0).toLocaleString();
  }

  function renderLoadError(err) {
    const tbody = document.querySelector('#data-tbody');
    if (!tbody) return;
    tbody.textContent = '';
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 15;
    td.className = 'table-empty';
    td.textContent = `Could not load sessions_raw.json — `
                   + `${err && err.message ? err.message : err}. `
                   + `Run \`python docs/build_dashboard.py\` first.`;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

})();
