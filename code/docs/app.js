/* =========================================================
 * Drishti-PAM dashboard — front-end logic
 * ─────────────────────────────────────────────────────────
 * दृष्टि — the gaze that sees beyond the surface.
 * =========================================================
 * Pure ES2020. No bundler, no build step. Loaded with `defer`
 * after the static HTML in index.html. Chart.js is loaded via
 * a CDN <script defer> tag in the same document; this file
 * waits on `window.Chart` before initialising the donut.
 *
 * The dashboard is a 2-tab single-page app:
 *   #/triage  Triage Console — what the detectors did with each
 *             session (default, owned by this file).
 *   #/data    Data Source    — the raw synthetic corpus, owned
 *             by data.js. data.js is told when its tab becomes
 *             visible via the `pam:data-view-shown` window event
 *             this file dispatches.
 *
 * Data sources (all produced by docs/build_dashboard.py):
 *   data/summary.json            small KPI + aggregate counts
 *   data/sessions_summary.json   per-session records (triage view)
 *   data/sessions_details.json   per-session drawer payload (lazy)
 *   data/sessions_raw.json       per-session inputs (data view, lazy)
 *   data/threat_model.json       STRIDE × ATT&CK × kill-chain mapping
 *                                (powers the Threat Coverage tile +
 *                                 the drawer's ATT&CK chips). Optional —
 *                                 the tile renders an empty state if the
 *                                 file is missing.
 *
 * The page never reaches out to anything else at runtime.
 * ========================================================= */

(() => {
  'use strict';

  // ----------------------------------------------------------
  //  State
  // ----------------------------------------------------------
  const state = {
    summary: null,
    sessions: [],
    threatModel: null,    // full data/threat_model.json payload
    filters: {
      risk_rating: '',
      compliance: '',
      scenario_gt: '',
      agreement: '',
      search: '',
    },
    sort: { col: 'risk_rating', dir: 'desc' },
    page: 1,
    pageSize: 50,
    visible: [],
    chart: null,           // risk-distribution donut
    scenarioChart: null,   // scenario × detector-agreement stacked bar
    // Drawer state. `details` lazily-fetched JSON keyed by
    // session id; `detailsLoading` is the in-flight Promise so
    // concurrent row clicks share one fetch.
    details: null,
    detailsLoading: null,
  };

  // Severity-driven sort orders. Larger value = more severe so the
  // natural DESC sort surfaces CRITICAL / non-compliant rows at the
  // top of the table — which is what an analyst expects.
  const RISK_ORDER = {
    CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1, MINIMAL: 0,
  };
  const COMPLIANCE_ORDER = {
    non_compliant: 2,
    partially_compliant: 1,
    fully_compliant: 0,
  };

  // Plain-English description per scenario ID. Mirrors the
  // generator's `_SCENARIO_EMITTERS` keys; if you ever add a
  // scenario, add it here too. The short label appears in the table
  // and filter dropdown; the longer description shows up as a hover
  // title so analysts get the full meaning without taking screen
  // real estate.
  const SCENARIO_INFO = {
    S1:  { short: 'Multi-user ticket',
           long:  'Several users sharing one change ticket within an hour.' },
    S2:  { short: 'Vague justification',
           long:  'Change explanation is too generic to defend.' },
    S3:  { short: 'Off-window session',
           long:  'Work happens before or after the approved change window.' },
    S4:  { short: 'Wrong-category commands',
           long:  "Commands don't match the ticket's stated category." },
    S5:  { short: 'Off-hours work',
           long:  'Session starts outside business hours.' },
    S7:  { short: 'Destructive on read-only',
           long:  'DELETE/DROP/TRUNCATE on a non-destructive ticket.' },
    S10: { short: 'Out-of-scope asset',
           long:  'Commands target a host not on the named-assets list.' },
    S11: { short: 'Zombie ticket',
           long:  'Same ticket reused over many days, often outside its window.' },
    S12: { short: 'Sensitive-data probing',
           long:  'Queries against PII columns or secret files.' },
    S13: { short: 'Lateral movement',
           long:  'Jumping or probing other hosts beyond the ticket scope.' },
    S14: { short: 'Privilege escalation',
           long:  'sudo/su without ticket authorisation.' },
    S15: { short: 'Approver-as-implementer',
           long:  'Requester equals approver — segregation-of-duties break.' },
    S16: { short: 'Vocabulary-cloak adversarial',
           long:  'Destructive SQL paraphrased (CALL purge_*, EXEC sp_purgehistory, ALTER … RENAME TO _orphaned_*) so it avoids the rule detector\u2019s literal trigger phrases. Adversarial control-evaluation scenario; rule detector cannot catch S16 by construction.' },
    clean: { short: 'No flagged scenario',
             long:  'No injected suspicious pattern (genuinely clean session).' },
  };

  // Publish on window so data.js (the second-tab owner) can
  // reuse the same plain-English vocabulary instead of keeping
  // its own copy in sync.
  window.PAM_SCENARIO_INFO = SCENARIO_INFO;

  // Risk colours (hex). Matches the CSS palette so the donut
  // and the table badges read as the same colour for the same
  // rating. Pulled from the design system, not from CSS vars,
  // because Chart.js needs a hex string.
  const RISK_COLOURS_DARK = {
    CRITICAL: '#ff2e63',
    HIGH:     '#ff4d6d',
    MEDIUM:   '#ffce5e',
    LOW:      '#56e0ff',
    MINIMAL:  '#5c6478',
  };
  const RISK_COLOURS_LIGHT = {
    CRITICAL: '#a40e2c',
    HIGH:     '#c8233f',
    MEDIUM:   '#b8771a',
    LOW:      '#1781a3',
    MINIMAL:  '#98a0b3',
  };

  // ----------------------------------------------------------
  //  Boot
  // ----------------------------------------------------------

  document.addEventListener('DOMContentLoaded', boot);

  async function boot() {
    document.body.classList.add('is-loading');
    wireTheme();
    wireFilters();
    wireSorting();
    wirePagination();
    wireDrawer();
    wireRouter();
    wireHeroCTAs();

    try {
      const [summary, sessions] = await Promise.all([
        fetch('data/summary.json',         {cache: 'no-cache'}).then(r => r.json()),
        fetch('data/sessions_summary.json',{cache: 'no-cache'}).then(r => r.json()),
      ]);
      state.summary = summary;
      state.sessions = sessions;
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      showLoadError(err);
      return;
    }

    // Threat model is optional — the page still works if it 404s. We
    // don't include it in the Promise.all above because we don't want
    // a missing threat_model.json to take the whole page down.
    try {
      const resp = await fetch('data/threat_model.json',
                               {cache: 'no-cache'});
      if (resp.ok) {
        state.threatModel = await resp.json();
      }
    } catch (err) {
      console.warn('threat_model.json not loaded:', err);
    }

    populateScenarioFilter();
    renderKPIs();
    renderThreatCoverage();
    recompute();
    maybeOpenExplainers();
    maybeOpenSessionFromQuery();
    document.body.classList.remove('is-loading');
  }

  // ----------------------------------------------------------
  //  Hash router (#/triage default, #/data for raw corpus)
  // ----------------------------------------------------------
  //
  // Three things keep this trivial on purpose:
  //   1. Only two routes — anything that isn't `#/data` is the
  //      default `#/triage` view. `#explain` (legacy bookmark)
  //      therefore falls through to triage as expected.
  //   2. `applyRoute` toggles a single `hidden` attribute on each
  //      view container; the hidden one is removed from layout
  //      via CSS so its DOM doesn't burn time.
  //   3. data.js is owner of the data view but is told *by us*
  //      when it becomes visible (via `pam:data-view-shown`),
  //      so it can lazy-fetch sessions_raw.json on first reveal.

  function currentRoute() {
    const h = window.location.hash || '';
    if (h.startsWith('#/data')) return 'data';
    return 'triage';
  }

  function applyRoute() {
    const route = currentRoute();
    const triageView = document.querySelector('.view--triage');
    const dataView   = document.querySelector('.view--data');
    if (!triageView || !dataView) return;

    const showData = route === 'data';
    triageView.hidden = showData;
    dataView.hidden   = !showData;

    document.querySelectorAll('.tab-link').forEach(a => {
      a.classList.toggle('is-active', a.dataset.tab === route);
    });

    document.body.dataset.view = route;

    if (showData) {
      window.dispatchEvent(new CustomEvent('pam:data-view-shown'));
    }
  }

  function wireRouter() {
    window.addEventListener('hashchange', applyRoute);
    applyRoute();
  }

  // ----------------------------------------------------------
  //  Hero CTAs
  // ----------------------------------------------------------
  //
  // The "Open the operator console ↓" tertiary CTA needs to do two
  // things: (1) ensure we're on the Triage Console route — if the
  // user clicked a Data-Source-tab link earlier and then scrolled
  // back up to the hero, they should land in the operator view, not
  // the data view; (2) scroll smoothly past the hero to the tab
  // strip so the operator console is visible right under the
  // sticky-pinned tabs. Without this handler the bare anchor href
  // either fires the SPA router (no scroll) or jumps abruptly to a
  // fragment ID (no view switch). Doing it in JS keeps the
  // behaviour explicit and the markup minimal.
  function wireHeroCTAs() {
    const consoleCta = document.querySelector('[data-link="console-cta"]');
    if (consoleCta) {
      consoleCta.addEventListener('click', (event) => {
        event.preventDefault();
        if (currentRoute() !== 'triage') {
          window.location.hash = '#/triage';
        }
        const target = document.querySelector('.tab-strip');
        if (target && typeof target.scrollIntoView === 'function') {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    }
  }

  // Bookmark-friendly: visiting `?session=sess-000000` auto-opens
  // the drawer for that session. Lets analysts share a direct
  // triage link.
  function maybeOpenSessionFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get('session');
    if (sid) openDrawer(sid);
  }

  // Bookmark-friendly: visiting `#explain` auto-opens both
  // methodology explainer cards on load. Useful for sharing a link
  // that lands on the algorithm description rather than the data.
  function maybeOpenExplainers() {
    if (window.location.hash === '#explain') {
      document.querySelectorAll('details.explainer-card')
        .forEach(d => { d.open = true; });
    }
  }

  // ----------------------------------------------------------
  //  Theme toggle
  // ----------------------------------------------------------

  function wireTheme() {
    const btn = document.querySelector('.theme-toggle');
    if (!btn) return;

    const stored = localStorage.getItem('pam-theme');
    if (stored === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
      btn.setAttribute('aria-pressed', 'true');
    }

    btn.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      btn.setAttribute('aria-pressed', String(next === 'light'));
      localStorage.setItem('pam-theme', next);
      // Re-render the donut + scenario bar so their hex colours
      // match the new theme.
      if (state.chart) renderDonut();
      if (state.scenarioChart) renderScenarioChart();
    });
  }

  // ----------------------------------------------------------
  //  Filters
  // ----------------------------------------------------------

  function wireFilters() {
    document.querySelectorAll('[data-filter]').forEach(el => {
      const key = el.dataset.filter;
      el.addEventListener('input', () => {
        state.filters[key] = el.value;
        state.page = 1;
        recompute();
      });
    });

    const clearBtn = document.querySelector('[data-action="clear-filters"]');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        Object.keys(state.filters).forEach(k => state.filters[k] = '');
        document.querySelectorAll('[data-filter]').forEach(el => el.value = '');
        state.page = 1;
        recompute();
      });
    }
  }

  function populateScenarioFilter() {
    const sel = document.getElementById('filter-scenario');
    if (!sel || !state.summary) return;
    const counts = state.summary.scenario_counts || {};
    const scenarios = Object.keys(counts).sort((a, b) => {
      // scenario IDs sorted numerically (S1, S2, ..., S16) with
      // 'clean' last.
      if (a === 'clean') return 1;
      if (b === 'clean') return -1;
      return parseInt(a.slice(1)) - parseInt(b.slice(1));
    });
    scenarios.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      const info = SCENARIO_INFO[s];
      const short = info ? info.short : s;
      // Show plain-English label first; the technical ID
      // follows in parentheses so analysts familiar with the
      // scenario library can still cross-reference. Count is
      // appended for triage scale.
      opt.textContent = s === 'clean'
        ? `${short} · ${counts[s]}`
        : `${short} (${s}) · ${counts[s]}`;
      sel.appendChild(opt);
    });
  }

  // ----------------------------------------------------------
  //  Sorting
  // ----------------------------------------------------------

  function wireSorting() {
    document.querySelectorAll('th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.sort;
        if (state.sort.col === col) {
          state.sort.dir = state.sort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          state.sort.col = col;
          state.sort.dir = 'asc';
        }
        renderTable();
        renderSortIndicators();
      });
    });
  }

  function renderSortIndicators() {
    document.querySelectorAll('th[data-sort]').forEach(th => {
      if (th.dataset.sort === state.sort.col) {
        th.dataset.sortActive = state.sort.dir;
      } else {
        delete th.dataset.sortActive;
      }
    });
  }

  // ----------------------------------------------------------
  //  Pagination
  // ----------------------------------------------------------

  function wirePagination() {
    const prev = document.querySelector('[data-action="prev-page"]');
    const next = document.querySelector('[data-action="next-page"]');
    prev?.addEventListener('click', () => {
      if (state.page > 1) { state.page--; renderTable(); }
    });
    next?.addEventListener('click', () => {
      const total = Math.max(1, Math.ceil(state.visible.length / state.pageSize));
      if (state.page < total) { state.page++; renderTable(); }
    });
  }

  // ----------------------------------------------------------
  //  Recompute pipeline (filter -> sort -> render)
  // ----------------------------------------------------------

  function recompute() {
    state.visible = applyFilters(state.sessions, state.filters);
    renderDonut();
    renderScenarioChart();
    renderTable();
    renderSortIndicators();
  }

  function applyFilters(rows, f) {
    const search = (f.search || '').trim().toLowerCase();
    return rows.filter(r => {
      if (f.risk_rating && r.risk_rating !== f.risk_rating) return false;
      if (f.compliance  && r.compliance  !== f.compliance)  return false;
      if (f.scenario_gt && r.scenario_gt !== f.scenario_gt) return false;
      if (f.agreement   && r.agreement   !== f.agreement)   return false;
      if (search) {
        const hay = `${r.user} ${r.host} ${r.ticket_id}`.toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    });
  }

  // ----------------------------------------------------------
  //  KPI rendering
  // ----------------------------------------------------------

  function renderKPIs() {
    const s = state.summary || {};

    // ----- Hero KPIs -----
    //
    // The hero section sits between the topbar and the tab strip, and
    // communicates the project to an executive reader in five seconds.
    // Three headline KPIs:
    //   1. Combined F1 (frozen reference run, 2,000-session test partition)
    //   2. Detector agreement (rule vs hybrid LLM, same call on test split)
    //   3. Analyst-hours saved per audit window (false-positive triage cost)
    //
    // Source of truth: summary.json::hero, populated by
    // build_dashboard.py::_build_hero_summary, which reads the
    // frozen reference comparison.json so the hero numbers stay
    // stable regardless of whether the local data dir holds a smoke
    // run. Falls back to dash placeholders if the reference file
    // is unavailable.
    const hero = s.hero || {};
    if (hero.available) {
      setBind('hero_f1_value', hero.f1_value);
      setBind('hero_f1_sub', hero.f1_sub);
      setBind('hero_agreement_value', hero.agreement_value);
      setBind('hero_agreement_sub', hero.agreement_sub);
      setBind('hero_opcost_value', hero.opcost_value);
      setBind('hero_opcost_sub', hero.opcost_sub);
    } else {
      setBind('hero_f1_value', '—');
      setBind('hero_f1_sub', 'reference data unavailable');
      setBind('hero_agreement_value', '—');
      setBind('hero_agreement_sub', 'reference data unavailable');
      setBind('hero_opcost_value', '—');
      setBind('hero_opcost_sub', 'reference data unavailable');
    }

    // ----- Operator console KPI strip (seven tiles below the hero) -----
    setBind('n_sessions',         fmtInt(s.n_sessions));
    setBind('n_risky',            fmtInt(s.n_risky));
    setBind('n_risky_pct',        s.n_risky_pct);
    setBind('n_noncompliant',     fmtInt(s.n_noncompliant));
    setBind('n_noncompliant_pct', s.n_noncompliant_pct);
    const topInfo = SCENARIO_INFO[s.top_scenario];
    setBind('top_scenario',
            topInfo ? topInfo.short : (s.top_scenario || '—'));
    setBind('top_scenario_id', s.top_scenario || '—');
    setBind('top_scenario_count', fmtInt(s.top_scenario_count));
    setBind('combined_f1',        (s.combined_f1 ?? 0).toFixed(4));
    setBind('combined_precision', (s.combined_precision ?? 0).toFixed(4));
    setBind('combined_recall',    (s.combined_recall ?? 0).toFixed(4));
    setBind('total_count',        fmtInt(s.n_sessions));
    setBind('generated_at',       formatGenAt(s.generated_at));
    // Deployment / containerisation tile. Source of truth:
    // build_dashboard.py inspects the Dockerfile + integrations/ folder
    // at build time and writes a deployment block into summary.json. The
    // tile is informational — it tells readers at a glance whether the
    // project ships SIEM/SOAR sketches and a runnable container, with
    // graceful fallback when those artefacts are absent (e.g., when the
    // dashboard is built from a sub-tree that doesn't include the repo
    // root's Dockerfile).
    const dep = s.deployment || {};
    if (dep.available) {
      setBind('deploy_status', 'Containerised');
      const ints = dep.integrations_count ?? 0;
      const intLabel = ints === 1 ? 'artefact' : 'artefacts';
      setBind(
        'deploy_sub',
        `${ints} SOAR/SIEM ${intLabel} \u00b7 Dockerfile ready`
      );
    } else {
      setBind('deploy_status', 'Local-only');
      setBind(
        'deploy_sub',
        'Dockerfile + integrations not detected'
      );
    }

    // ----- Operational FP triage cost -----
    //
    // Picks the *cheaper* of the two detectors as the "best-case" cost
    // a SOC would pay if it deployed only that detector. The sub-line
    // names which detector that is and the headline 18-min baseline.
    // Falls back to "—" when the operational_metrics block is absent
    // or both detectors have zero false positives.
    const op = s.operational_metrics || {};
    if (op.available && op.best_detector) {
      setBind('opcost_value', `${op.best_hours.toFixed(1)} h`);
      setBind(
        'opcost_sub',
        `${op.best_detector}: ${op.best_fp} FP \u00d7 ${op.fp_minutes} min`
      );
    } else {
      setBind('opcost_value', '\u2014');
      setBind(
        'opcost_sub',
        'operational metric \u2014 awaiting data'
      );
    }
  }

  // ----------------------------------------------------------
  //  Threat Coverage tile (static — does not react to filters)
  // ----------------------------------------------------------
  //
  // Inputs:
  //   data/summary.json -> .threat_coverage  (counts + top techniques,
  //                                           always present; .available
  //                                           is false when source JSON
  //                                           was missing at build time)
  //   data/threat_model.json                  (full per-scenario mapping;
  //                                           used by the drawer's chips
  //                                           but not strictly required
  //                                           for the tile itself)

  function renderThreatCoverage() {
    const s = state.summary || {};
    const cov = s.threat_coverage || {};
    const available = cov.available !== false
                       && (cov.unique_techniques_covered || 0) > 0;

    setBind('attck_version', cov.attck_version || 'v15.1');

    renderThreatStats(cov, available);
    renderThreatTechList(cov, available);
  }

  function renderThreatStats(cov, available) {
    const ul = document.getElementById('threat-stats');
    if (!ul) return;
    ul.innerHTML = '';

    if (!available) {
      // Render four empty placeholders so the layout doesn't shift
      // when the file is missing — the empty-notice next to the
      // technique list is what tells the user what to do.
      [
        ['Techniques',     '—'],
        ['Tactics',        '—'],
        ['Kill-chain',     '—'],
        ['Governance-only','—'],
      ].forEach(([label, val]) => ul.appendChild(threatStatLi(label, val)));
      return;
    }

    const conf = cov.confidence_distribution || {};
    const stride = (cov.stride_categories_present || []).join(' · ');
    const total = cov.total_scenarios || 0;
    const govCount = cov.governance_only_count || 0;

    ul.appendChild(threatStatLi(
      'ATT&CK techniques',
      String(cov.unique_techniques_covered || 0),
      `across ${total} scenarios`));
    ul.appendChild(threatStatLi(
      'ATT&CK tactics',
      String(cov.unique_tactics_covered || 0),
      'of 14 enterprise tactics'));
    ul.appendChild(threatStatLi(
      'Kill-chain phases',
      String(cov.kill_chain_phases_covered || 0),
      'unique phases touched'));
    ul.appendChild(threatStatLi(
      'STRIDE categories',
      String((cov.stride_categories_present || []).length),
      stride || 'none recorded'));
    ul.appendChild(threatStatLi(
      'Mapping confidence',
      `${conf.high || 0}H · ${conf.medium || 0}M · ${conf.low || 0}L`,
      'high / medium / low'));
    ul.appendChild(threatStatLi(
      'Governance-only',
      `${govCount} of ${total}`,
      (cov.governance_only_ids || []).join(', ') || '—'));
  }

  function threatStatLi(label, value, sub) {
    const li = document.createElement('li');
    const labWrap = document.createElement('div');
    const lab = document.createElement('span');
    lab.className = 'threat-stat-label';
    lab.textContent = label;
    labWrap.appendChild(lab);
    if (sub) {
      const subEl = document.createElement('span');
      subEl.className = 'threat-stat-sub';
      subEl.textContent = sub;
      labWrap.appendChild(subEl);
    }
    const val = document.createElement('span');
    val.className = 'threat-stat-value';
    val.textContent = value;
    li.append(labWrap, val);
    return li;
  }

  function renderThreatTechList(cov, available) {
    const ul = document.getElementById('threat-tech-list');
    const empty = document.getElementById('threat-empty');
    if (!ul) return;
    ul.innerHTML = '';

    if (!available) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    const techs = cov.top_techniques || [];
    if (techs.length === 0) {
      const li = document.createElement('li');
      li.style.color = 'var(--fg-2)';
      li.textContent = 'No techniques recorded.';
      ul.appendChild(li);
      return;
    }

    const max = Math.max(1, ...techs.map(t => t.scenario_count || 0));
    techs.forEach(t => {
      const li = document.createElement('li');

      const tag = document.createElement('span');
      tag.className = 'threat-tech-tag';
      tag.textContent = t.id || '?';

      const name = document.createElement('span');
      name.className = 'threat-tech-name';
      name.textContent = t.name || '?';
      name.title = `${t.id} · ${t.name}`;

      const bar = document.createElement('span');
      bar.className = 'threat-tech-bar';
      const fill = document.createElement('span');
      fill.className = 'threat-tech-bar-fill';
      fill.style.width =
        `${Math.round(100 * (t.scenario_count || 0) / max)}%`;
      bar.appendChild(fill);

      const cnt = document.createElement('span');
      cnt.className = 'threat-tech-count';
      cnt.textContent = `${t.scenario_count || 0}×`;
      cnt.title = `${t.scenario_count || 0} scenarios touch this technique`;

      li.append(tag, name, bar, cnt);
      ul.appendChild(li);
    });
  }

  // ----------------------------------------------------------
  //  Donut
  // ----------------------------------------------------------

  function renderDonut() {
    if (!window.Chart) {
      // Chart.js still loading — wait one tick and retry.
      window.setTimeout(renderDonut, 80);
      return;
    }
    const counts = countByRiskRating(state.visible);
    const labels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'];
    const data = labels.map(k => counts[k] || 0);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const palette = isLight ? RISK_COLOURS_LIGHT : RISK_COLOURS_DARK;
    const colours = labels.map(k => palette[k]);
    const fgMute = isLight ? '#555c6e' : '#98a0b3';
    const surface = isLight ? '#ffffff' : '#11151f';

    const canvas = document.getElementById('risk-donut');
    if (!canvas) return;

    if (state.chart) {
      state.chart.destroy();
    }

    state.chart = new window.Chart(canvas, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: colours,
          borderColor: surface,
          borderWidth: 3,
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        cutout: '62%',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: surface,
            titleColor: isLight ? '#1a1d27' : '#e6e8ee',
            bodyColor: fgMute,
            borderColor: isLight ? '#e1e4ec' : '#1f2535',
            borderWidth: 1,
            padding: 10,
            displayColors: true,
            callbacks: {
              label: (ctx) => {
                const total = data.reduce((a, b) => a + b, 0) || 1;
                const v = ctx.parsed;
                const pct = (100 * v / total).toFixed(1);
                return ` ${ctx.label}: ${v} (${pct}%)`;
              },
            },
          },
        },
      },
    });

    renderDonutLegend(labels, data, colours);
  }

  function renderDonutLegend(labels, data, colours) {
    const ul = document.getElementById('risk-legend');
    if (!ul) return;
    ul.innerHTML = '';
    const total = data.reduce((a, b) => a + b, 0) || 1;
    labels.forEach((label, i) => {
      const li = document.createElement('li');
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = colours[i];
      const lab = document.createElement('span');
      lab.className = 'legend-label';
      lab.textContent = label;
      const cnt = document.createElement('span');
      cnt.className = 'legend-count';
      const pct = (100 * data[i] / total).toFixed(1);
      cnt.textContent = `${data[i]} · ${pct}%`;
      li.append(swatch, lab, cnt);
      ul.appendChild(li);
    });
  }

  function countByRiskRating(rows) {
    const out = {};
    rows.forEach(r => out[r.risk_rating] = (out[r.risk_rating] || 0) + 1);
    return out;
  }

  // ----------------------------------------------------------
  //  Scenario × detector-agreement stacked bar
  // ----------------------------------------------------------

  function renderScenarioChart() {
    if (!window.Chart) {
      window.setTimeout(renderScenarioChart, 80);
      return;
    }

    // Group visible sessions by ground-truth scenario, then by
    // detector-agreement bucket. The chart respects the active
    // filters (so picking 'Non-compliant' in the filter panel
    // narrows the bars to only those sessions, etc.).
    //
    // We deliberately exclude `clean` ground-truth sessions: on
    // realistic data they outnumber suspicious ones ~30:1 and
    // would dominate the y-axis, flattening the actually-
    // interesting bars. The donut + KPIs already surface the
    // clean fraction.
    const buckets = {};
    state.visible.forEach(r => {
      const sid = r.scenario_gt;
      if (!sid || sid === 'clean') return;
      if (!buckets[sid]) {
        buckets[sid] = {
          both_flag: 0,
          rule_only_flag: 0,
          llm_only_flag: 0,
          both_clean: 0,
        };
      }
      buckets[sid][r.agreement] += 1;
    });

    // Sort scenarios numerically (S1, S2, …).
    const scenarios = Object.keys(buckets).sort((a, b) => {
      return parseInt(a.slice(1)) - parseInt(b.slice(1));
    });

    const labels = scenarios.map(s => {
      const info = SCENARIO_INFO[s];
      return info ? info.short : s;
    });

    // No suspicious scenarios in view (e.g. user filtered to
    // Risk = MINIMAL only). Show an empty-state instead of an
    // empty chart so analysts know it's a filter, not a bug.
    const wrap = document.getElementById('scenario-chart')?.parentElement;
    if (wrap) {
      const existingEmpty = wrap.querySelector('.chart-empty');
      if (existingEmpty) existingEmpty.remove();
      if (scenarios.length === 0) {
        if (state.scenarioChart) {
          state.scenarioChart.destroy();
          state.scenarioChart = null;
        }
        const empty = document.createElement('div');
        empty.className = 'chart-empty';
        empty.textContent = 'No suspicious scenarios match the current filters.';
        wrap.appendChild(empty);
        return;
      }
    }

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const fgMute   = isLight ? '#555c6e' : '#98a0b3';
    const surface  = isLight ? '#ffffff' : '#11151f';
    const lineCol  = isLight ? '#e1e4ec' : '#1f2535';

    // Stack colours match the detector-agreement semantics:
    //   both_flag      → red       (both detectors say bad)
    //   rule_only_flag → orange    (only rule caught it)
    //   llm_only_flag  → yellow    (only LLM caught it)
    //   both_clean     → grey      (both detectors say clean)
    const stack = isLight
      ? { both_flag: '#c8233f',
          rule_only_flag: '#c64a1c',
          llm_only_flag: '#b8771a',
          both_clean: '#98a0b3' }
      : { both_flag: '#ff4d6d',
          rule_only_flag: '#ff9e5e',
          llm_only_flag: '#ffce5e',
          both_clean: '#5c6478' };

    const datasets = [
      { key: 'both_flag',      label: 'Both flag' },
      { key: 'rule_only_flag', label: 'Rule only' },
      { key: 'llm_only_flag',  label: 'LLM only' },
      { key: 'both_clean',     label: 'Both clean' },
    ].map(d => ({
      label: d.label,
      data: scenarios.map(s => buckets[s][d.key]),
      backgroundColor: stack[d.key],
      borderColor: surface,
      borderWidth: 1,
    }));

    const canvas = document.getElementById('scenario-chart');
    if (!canvas) return;

    if (state.scenarioChart) {
      state.scenarioChart.destroy();
    }

    state.scenarioChart = new window.Chart(canvas, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 4, right: 8 } },
        scales: {
          x: {
            stacked: true,
            ticks: {
              color: fgMute,
              font: { size: 11 },
              maxRotation: 38,
              minRotation: 32,
              autoSkip: false,
            },
            grid: { display: false, drawBorder: false },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            ticks: {
              color: fgMute,
              font: { size: 11 },
              precision: 0,
            },
            grid: { color: lineCol, drawBorder: false },
          },
        },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              color: fgMute,
              font: { size: 11 },
              boxWidth: 12,
              boxHeight: 12,
              padding: 14,
              usePointStyle: false,
            },
          },
          tooltip: {
            backgroundColor: surface,
            titleColor: isLight ? '#1a1d27' : '#e6e8ee',
            bodyColor: fgMute,
            borderColor: lineCol,
            borderWidth: 1,
            padding: 10,
            callbacks: {
              title: (items) => {
                const idx = items[0].dataIndex;
                const sid = scenarios[idx];
                const info = SCENARIO_INFO[sid];
                if (!info) return sid;
                return sid === 'clean'
                  ? info.short
                  : `${info.short} (${sid})`;
              },
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}`,
              footer: (items) => {
                const total = items.reduce((acc, x) => acc + x.parsed.y, 0);
                return `Total: ${total}`;
              },
            },
          },
        },
      },
    });
  }

  // ----------------------------------------------------------
  //  Table rendering
  // ----------------------------------------------------------

  function renderTable() {
    const tbody = document.getElementById('session-tbody');
    if (!tbody) return;

    const sorted = sortRows(state.visible, state.sort);
    const total = sorted.length;
    const totalPages = Math.max(1, Math.ceil(total / state.pageSize));
    if (state.page > totalPages) state.page = totalPages;
    const startIdx = (state.page - 1) * state.pageSize;
    const pageRows = sorted.slice(startIdx, startIdx + state.pageSize);

    setBind('filtered_count', fmtInt(total));
    setBind('page_current', state.page);
    setBind('page_total', totalPages);

    document.querySelector('[data-action="prev-page"]').disabled = state.page <= 1;
    document.querySelector('[data-action="next-page"]').disabled = state.page >= totalPages;

    tbody.innerHTML = '';
    if (pageRows.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 8;
      td.style.textAlign = 'center';
      td.style.padding = '2rem 0';
      td.style.color = 'var(--fg-1)';
      td.textContent = total === 0 && state.sessions.length === 0
        ? 'No sessions loaded.'
        : 'No sessions match the current filters.';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    const frag = document.createDocumentFragment();
    pageRows.forEach(r => frag.appendChild(buildRow(r)));
    tbody.appendChild(frag);
  }

  function sortRows(rows, sort) {
    const dir = sort.dir === 'desc' ? -1 : 1;
    const col = sort.col;
    const sorted = rows.slice();
    sorted.sort((a, b) => dir * compareForCol(a, b, col));
    return sorted;
  }

  function compareForCol(a, b, col) {
    switch (col) {
      case 'risk_rating':
        return (RISK_ORDER[a.risk_rating] ?? 9) - (RISK_ORDER[b.risk_rating] ?? 9);
      case 'compliance':
        return (COMPLIANCE_ORDER[a.compliance] ?? 9) - (COMPLIANCE_ORDER[b.compliance] ?? 9);
      case 'llm_score':
        return (a.llm_score ?? 0) - (b.llm_score ?? 0);
      default: {
        const av = (a[col] ?? '').toString().toLowerCase();
        const bv = (b[col] ?? '').toString().toLowerCase();
        if (av < bv) return -1;
        if (av > bv) return 1;
        return 0;
      }
    }
  }

  function buildRow(r) {
    const tr = document.createElement('tr');
    tr.dataset.sessionId = r.id;
    tr.tabIndex = 0;
    tr.setAttribute('role', 'button');
    tr.setAttribute('aria-label',
      `Open details for session ${r.id}`);

    tr.appendChild(td(r.id, 'mono'));
    tr.appendChild(td(spanTrunc(r.user), 'truncate-cell'));
    tr.appendChild(td(spanTrunc(r.host, 'truncate-cell--wide'), 'truncate-cell--wide'));
    tr.appendChild(td(scenarioCell(r), 'scenario-cell ' + scenarioCellMod(r)));
    tr.appendChild(td(callCell(r.rule_call), 'call-cell ' + callCellMod(r.rule_call)));
    tr.appendChild(td(scoreCell(r), ''));
    tr.appendChild(td(riskBadge(r.risk_rating), ''));
    tr.appendChild(td(compliancePill(r.compliance, r.compliance_pass_count), ''));

    return tr;
  }

  function td(child, cls) {
    const el = document.createElement('td');
    if (cls) el.className = cls;
    if (typeof child === 'string') el.textContent = child;
    else if (child instanceof Node) el.appendChild(child);
    return el;
  }

  function spanTrunc(text, extraCls) {
    const s = document.createElement('span');
    s.className = 'truncate-cell' + (extraCls ? ' ' + extraCls : '');
    s.textContent = text || '—';
    s.title = text || '';
    return s;
  }

  function scenarioCell(r) {
    // Render as a plain-English label with the technical ID tucked
    // into a small monospace tag. Long description goes in the title
    // attribute so hovering reveals the full meaning.
    const sid = r.scenario_gt || 'clean';
    const info = SCENARIO_INFO[sid] || { short: sid, long: '' };

    const wrap = document.createElement('span');
    wrap.className = 'scen';
    wrap.title = info.long;

    const label = document.createElement('span');
    label.className = 'scen-label';
    label.textContent = info.short;

    wrap.appendChild(label);

    if (sid !== 'clean') {
      const tag = document.createElement('span');
      tag.className = 'scen-tag';
      tag.textContent = sid;
      wrap.appendChild(tag);
    }

    return wrap;
  }
  function scenarioCellMod(r) {
    return r.label_gt === 'suspicious'
      ? 'scenario-cell--suspicious'
      : 'scenario-cell--clean';
  }

  function callCell(call) {
    return call === 'suspicious' ? 'SUSP' : 'clean';
  }
  function callCellMod(call) {
    return call === 'suspicious'
      ? 'call-cell--suspicious'
      : 'call-cell--clean';
  }

  function scoreCell(r) {
    const wrap = document.createElement('span');
    wrap.className = 'score-cell';

    const bar = document.createElement('span');
    bar.className = 'score-bar';
    const fill = document.createElement('span');
    fill.className = 'score-bar-fill';
    fill.style.width = `${Math.round(100 * (r.llm_score ?? 0))}%`;
    bar.appendChild(fill);

    const num = document.createElement('span');
    num.className = 'score-num';
    num.textContent = (r.llm_score ?? 0).toFixed(2);

    wrap.append(bar, num);
    return wrap;
  }

  function riskBadge(rating) {
    const span = document.createElement('span');
    const cls = (rating || 'minimal').toLowerCase();
    span.className = `badge badge--${cls}`;
    span.textContent = rating;
    return span;
  }

  function compliancePill(label, pass) {
    const span = document.createElement('span');
    const map = {
      fully_compliant:    { mod: 'pill--full',    text: 'Compliant' },
      partially_compliant:{ mod: 'pill--partial', text: 'Partial'   },
      non_compliant:      { mod: 'pill--non',     text: 'Non'       },
    };
    const m = map[label] || { mod: '', text: '—' };
    span.className = `pill ${m.mod}`;
    span.textContent = `${m.text} · ${pass ?? 0}/5`;
    span.title = `${pass ?? 0} of 5 policy signals pass`;
    return span;
  }

  // ----------------------------------------------------------
  //  Helpers
  // ----------------------------------------------------------

  function setBind(key, value) {
    document.querySelectorAll(`[data-bind="${key}"]`).forEach(el => {
      el.textContent = value === null || value === undefined ? '—' : String(value);
    });
  }

  function fmtInt(n) {
    if (n === null || n === undefined) return '—';
    return Number(n).toLocaleString('en-US');
  }

  function formatGenAt(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      const yyyy = d.getUTCFullYear();
      const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(d.getUTCDate()).padStart(2, '0');
      const hh = String(d.getUTCHours()).padStart(2, '0');
      const mi = String(d.getUTCMinutes()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd} ${hh}:${mi} UTC`;
    } catch { return iso; }
  }

  // ----------------------------------------------------------
  //  Drawer (session detail)
  // ----------------------------------------------------------

  function wireDrawer() {
    // Row click — delegated on the table body so re-renders never
    // need to re-attach listeners.
    const tbody = document.getElementById('session-tbody');
    if (tbody) {
      tbody.addEventListener('click', (ev) => {
        const tr = ev.target.closest('tr[data-session-id]');
        if (tr) openDrawer(tr.dataset.sessionId);
      });
      // Keyboard: Enter / Space on a focused row opens the drawer.
      tbody.addEventListener('keydown', (ev) => {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        const tr = ev.target.closest('tr[data-session-id]');
        if (tr) {
          ev.preventDefault();
          openDrawer(tr.dataset.sessionId);
        }
      });
    }

    // Close button + overlay click.
    document.querySelectorAll('[data-action="close-drawer"]')
      .forEach(el => el.addEventListener('click', closeDrawer));

    // ESC key closes the drawer when it's open.
    document.addEventListener('keydown', (ev) => {
      if (ev.key !== 'Escape') return;
      const drawer = document.getElementById('session-drawer');
      if (drawer && drawer.getAttribute('aria-hidden') === 'false') {
        closeDrawer();
      }
    });
  }

  async function openDrawer(sessionId) {
    const drawer = document.getElementById('session-drawer');
    const body = document.getElementById('drawer-body');
    if (!drawer || !body) return;

    // Reset header bindings to a placeholder while the details
    // load. Fast-path: if the details map is already cached we
    // skip straight to render.
    setDrawerHeaderPlaceholder(sessionId);
    drawer.setAttribute('aria-hidden', 'false');
    document.body.classList.add('drawer-open');

    if (!state.details) {
      body.innerHTML =
        '<p class="drawer-loading">Loading session details…</p>';
      try {
        if (!state.detailsLoading) {
          state.detailsLoading = fetch(
            'data/sessions_details.json', {cache: 'no-cache'}
          ).then(r => r.json());
        }
        state.details = await state.detailsLoading;
      } catch (err) {
        console.error('Failed to load session details:', err);
        body.innerHTML =
          '<p class="drawer-loading" style="color:var(--bad)">' +
          'Failed to load <code>data/sessions_details.json</code>. ' +
          'Re-run <code>python docs/build_dashboard.py</code> from ' +
          '<code>code/</code>.</p>';
        return;
      }
    }

    const detail = state.details[sessionId];
    if (!detail) {
      body.innerHTML =
        '<p class="drawer-loading" style="color:var(--bad)">' +
        `Session <code>${sessionId}</code> not found in the details ` +
        'file. The summary and details artefacts may be out of sync.' +
        '</p>';
      return;
    }
    renderDrawer(detail);
  }

  function closeDrawer() {
    const drawer = document.getElementById('session-drawer');
    if (!drawer) return;
    drawer.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('drawer-open');
  }

  function setDrawerHeaderPlaceholder(sessionId) {
    const title = document.getElementById('drawer-title');
    const eyebrow = document.getElementById('drawer-eyebrow');
    if (title) title.textContent = sessionId;
    if (eyebrow) eyebrow.textContent = 'SESSION';
    document.getElementById('drawer-risk').textContent = '—';
    document.getElementById('drawer-risk').className = 'badge badge--minimal';
    document.getElementById('drawer-compliance').textContent = '—';
    document.getElementById('drawer-compliance').className = 'pill';
    document.getElementById('drawer-meta-sub').textContent = '';
  }

  function renderDrawer(d) {
    // Header
    const title = document.getElementById('drawer-title');
    title.textContent = d.id;

    const risk = document.getElementById('drawer-risk');
    risk.className = 'badge badge--' + (d.risk_rating || 'minimal').toLowerCase();
    risk.textContent = d.risk_rating;

    const comp = document.getElementById('drawer-compliance');
    const compMap = {
      fully_compliant:    { mod: 'pill--full',    text: 'Compliant' },
      partially_compliant:{ mod: 'pill--partial', text: 'Partial'   },
      non_compliant:      { mod: 'pill--non',     text: 'Non-compliant' },
    };
    const cm = compMap[d.compliance] || { mod: '', text: '—' };
    comp.className = `pill ${cm.mod}`;
    comp.textContent = `${cm.text} · ${d.compliance_pass_count}/5`;

    const sub = document.getElementById('drawer-meta-sub');
    sub.textContent = `${d.user} · ${d.host}`;

    // Body
    const body = document.getElementById('drawer-body');
    body.innerHTML = '';
    body.appendChild(renderOverviewSection(d));
    body.appendChild(renderVerdictSection(d));
    body.appendChild(renderGroundTruthSection(d));
    body.appendChild(renderComplianceSection(d));
    body.appendChild(renderJustificationSection(d));
    body.appendChild(renderCommandStreamSection(d));
    if (d.vault_fetches && d.vault_fetches.length) {
      body.appendChild(renderVaultFetchesSection(d));
    }
  }

  // ---- Section builders ------------------------------------

  function section(title) {
    const s = document.createElement('section');
    s.className = 'drawer-section';
    const h = document.createElement('h3');
    h.className = 'drawer-section-title';
    h.textContent = title;
    s.appendChild(h);
    return s;
  }

  function kvRow(parent, key, value) {
    const dl = parent.querySelector('dl.kv-grid')
      || (() => {
        const d = document.createElement('dl');
        d.className = 'kv-grid';
        parent.appendChild(d);
        return d;
      })();
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    if (value instanceof Node) dd.appendChild(value);
    else dd.textContent = value;
    dl.append(dt, dd);
  }

  function renderOverviewSection(d) {
    const sec = section('Overview');
    kvRow(sec, 'User', d.user);
    kvRow(sec, 'Asset', d.host);
    kvRow(sec, 'Ticket', `${d.ticket_id} · ${d.ticket_category}`);
    kvRow(sec, 'Approver', d.ticket_approver);
    kvRow(sec, 'Started', formatTimestamp(d.start_time));
    kvRow(sec, 'Window',
          `${formatTimestamp(d.change_window_start)} → ${formatTimestamp(d.change_window_end)}`);
    if (d.named_assets && d.named_assets.length) {
      kvRow(sec, 'Named assets', d.named_assets.join(', '));
    }
    kvRow(sec, 'Commands', `${d.command_count_total}`);
    return sec;
  }

  function renderVerdictSection(d) {
    const sec = section('Detector verdicts');
    const row = document.createElement('div');
    row.className = 'verdict-row';

    // Rule card
    const rule = document.createElement('article');
    rule.className = 'verdict-card';
    const rTitle = document.createElement('h4');
    rTitle.className = 'verdict-card-title';
    rTitle.textContent = 'Rules detector';
    rule.appendChild(rTitle);

    const rCall = document.createElement('p');
    rCall.className = 'verdict-call ' +
      (d.rule_call === 'suspicious'
        ? 'verdict-call--suspicious'
        : 'verdict-call--clean');
    rCall.textContent = (d.rule_call || 'clean').toUpperCase();
    rule.appendChild(rCall);

    const rDetail = document.createElement('p');
    rDetail.className = 'verdict-detail';
    if ((d.rule_fires || []).length) {
      rDetail.innerHTML = `<strong>Rules fired:</strong> ${d.rule_fires.length}`;
    } else {
      rDetail.innerHTML = '<strong>Rules fired:</strong> none';
    }
    rule.appendChild(rDetail);

    if ((d.rule_fires || []).length) {
      const ul = document.createElement('ul');
      ul.className = 'verdict-rules';
      d.rule_fires.forEach(sid => {
        const li = document.createElement('li');
        const tag = document.createElement('span');
        tag.className = 'verdict-rule-tag';
        tag.textContent = sid;
        const name = document.createElement('span');
        name.className = 'verdict-rule-name';
        const info = SCENARIO_INFO[sid] || { short: sid, long: '' };
        name.textContent = info.short;
        li.append(tag, name);
        if (info.long) {
          const desc = document.createElement('span');
          desc.className = 'verdict-rule-desc';
          desc.textContent = info.long;
          li.appendChild(desc);
        }
        ul.appendChild(li);
      });
      rule.appendChild(ul);
    }

    if (d.rule_evidence) {
      const ev = document.createElement('div');
      ev.className = 'evidence-block';
      ev.textContent = d.rule_evidence;
      rule.appendChild(ev);
    }

    row.appendChild(rule);

    // LLM card
    const llm = document.createElement('article');
    llm.className = 'verdict-card';
    const lTitle = document.createElement('h4');
    lTitle.className = 'verdict-card-title';
    lTitle.textContent = 'LLM detector';
    llm.appendChild(lTitle);

    const lCall = document.createElement('p');
    lCall.className = 'verdict-call ' +
      (d.llm_call === 'suspicious'
        ? 'verdict-call--suspicious'
        : 'verdict-call--clean');
    lCall.textContent = (d.llm_call || 'clean').toUpperCase();
    llm.appendChild(lCall);

    const lDetail = document.createElement('p');
    lDetail.className = 'verdict-detail';
    const conf = (d.llm_score ?? 0).toFixed(3);
    const confLabel = describeLlmConfidence(d.llm_score, d.llm_call);
    lDetail.innerHTML =
      `<strong>p(suspicious)</strong> = ${conf} <em>(${confLabel})</em>`;
    llm.appendChild(lDetail);

    const sr = document.createElement('div');
    sr.className = 'score-readout';
    const bar = document.createElement('div');
    bar.className = 'score-readout-bar';
    const fill = document.createElement('div');
    fill.className = 'score-readout-fill';
    fill.style.width = `${Math.round(100 * (d.llm_score ?? 0))}%`;
    bar.appendChild(fill);
    const num = document.createElement('span');
    num.className = 'score-readout-num';
    num.textContent = (d.llm_score ?? 0).toFixed(2);
    sr.append(bar, num);
    llm.appendChild(sr);

    if (d.llm_evidence) {
      const ev = document.createElement('div');
      ev.className = 'evidence-block';
      ev.textContent = d.llm_evidence;
      llm.appendChild(ev);
    }

    row.appendChild(llm);
    sec.appendChild(row);
    return sec;
  }

  function renderGroundTruthSection(d) {
    const sec = section('Ground truth');
    const sid = d.scenario_gt || 'clean';
    const info = SCENARIO_INFO[sid] || { short: sid, long: '' };
    const banner = document.createElement('div');
    banner.className = 'gt-banner';

    const head = document.createElement('div');
    if (sid === 'clean') {
      head.innerHTML = `<span class="gt-banner-title">${info.short}</span>`;
    } else {
      head.innerHTML =
        `<span class="gt-banner-tag">${sid}</span>` +
        `<span class="gt-banner-title">${info.short}</span>`;
    }
    banner.appendChild(head);

    if (info.long) {
      const desc = document.createElement('span');
      desc.className = 'gt-banner-desc';
      desc.textContent = info.long;
      banner.appendChild(desc);
    }

    const verdict = document.createElement('p');
    verdict.className = 'gt-banner-verdict';
    verdict.textContent = d.label_gt === 'suspicious'
      ? 'This session is genuinely suspicious — the synthetic generator injected the scenario above.'
      : 'This session is genuinely clean — no suspicious pattern was injected by the generator.';
    banner.appendChild(verdict);

    // ATT&CK chips. Only shown when the threat-model JSON loaded AND
    // the ground-truth scenario is non-clean AND has at least one
    // mapped technique. The chip set is intentionally compact — full
    // tactic / kill-chain context lives in the Threat Coverage tile.
    if (sid !== 'clean' && state.threatModel) {
      const sdata = (state.threatModel.scenarios || {})[sid];
      const techs = (sdata && sdata.attck && sdata.attck.techniques) || [];
      if (techs.length > 0) {
        const wrap = document.createElement('div');
        wrap.className = 'gt-attck';
        techs.forEach(t => {
          const tag = document.createElement('span');
          tag.className = 'gt-attck-tag';
          tag.title = `${t.id} · ${t.name}`;
          const idEl = document.createElement('span');
          idEl.textContent = t.id || '?';
          const nameEl = document.createElement('span');
          nameEl.className = 'gt-attck-tag-name';
          nameEl.textContent = t.name || '';
          tag.append(idEl, nameEl);
          wrap.appendChild(tag);
        });
        banner.appendChild(wrap);
      }
    }

    sec.appendChild(banner);
    return sec;
  }

  function renderComplianceSection(d) {
    const sec = section('Policy compliance');
    const ul = document.createElement('ul');
    ul.className = 'signal-list';
    const sigs = d.compliance_signals || {};
    [
      ['s1_justification_length',           'Justification length'],
      ['s2_command_count',                  'Command count'],
      ['s3_business_hours',                 'Business hours'],
      ['s4_sensitive_keywords',             'No sensitive keywords'],
      ['s5_justification_command_overlap',  'Justification ↔ command overlap'],
    ].forEach(([key, fallback]) => {
      const sig = sigs[key];
      if (!sig) return;
      const li = document.createElement('li');

      const mark = document.createElement('span');
      mark.className = 'signal-mark ' +
        (sig.passed ? 'signal-mark--pass' : 'signal-mark--fail');
      mark.textContent = sig.passed ? '✓' : '✗';

      const right = document.createElement('div');
      const name = document.createElement('span');
      name.className = 'signal-name';
      name.textContent = sig.label || fallback;
      right.appendChild(name);

      const detail = document.createElement('span');
      detail.className = 'signal-detail';
      detail.textContent = describeComplianceSignal(key, sig);
      right.appendChild(detail);

      li.append(mark, right);
      ul.appendChild(li);
    });
    sec.appendChild(ul);
    return sec;
  }

  function describeComplianceSignal(key, sig) {
    switch (key) {
      case 's1_justification_length':
        return `${sig.value} word${sig.value === 1 ? '' : 's'}` +
               ` (threshold ≥ ${sig.threshold})`;
      case 's2_command_count':
        return `${sig.value} command${sig.value === 1 ? '' : 's'}` +
               ` (threshold ≤ ${sig.threshold})`;
      case 's3_business_hours':
        // The Python builder pre-formats this as a wall-clock
        // string. We never re-parse it via `new Date(...)` because
        // the synthetic timestamps have no TZ marker and JS would
        // helpfully convert them to the browser's local time.
        return sig.display || sig.value || 'No timestamp recorded';
      case 's4_sensitive_keywords':
        if (sig.passed) return 'Command stream is clean';
        return 'Sensitive terms in commands: ' +
               (sig.offenders || []).join(', ');
      case 's5_justification_command_overlap':
        if (sig.passed && (sig.shared_terms || []).length) {
          return 'Shared keywords: ' + sig.shared_terms.slice(0, 4).join(', ');
        }
        return sig.passed ? 'At least one shared keyword' : 'No shared keywords';
      default:
        return '';
    }
  }

  function renderJustificationSection(d) {
    const sec = section('Business justification');
    const block = document.createElement('blockquote');
    block.className = 'justification-block';
    block.textContent = d.justification || '(empty)';
    sec.appendChild(block);
    return sec;
  }

  function renderCommandStreamSection(d) {
    const total = d.command_count_total ?? 0;
    const shown = d.commands ? d.commands.length : 0;
    const sec = section(`Command stream (${total})`);

    if (shown === 0) {
      const p = document.createElement('p');
      p.className = 'verdict-detail';
      p.textContent = 'No commands recorded for this session.';
      sec.appendChild(p);
      return sec;
    }

    const ul = document.createElement('ul');
    ul.className = 'command-list';
    d.commands.forEach(c => {
      const li = document.createElement('li');
      const ts = document.createElement('span');
      ts.className = 'command-ts';
      ts.textContent = (c.ts || '').slice(11, 19) || '—';
      const surf = document.createElement('span');
      surf.className = 'command-surface';
      surf.textContent = c.surface || '—';
      const text = document.createElement('span');
      text.className = 'command-text';
      text.textContent = c.command || '';
      li.append(ts, surf, text);
      ul.appendChild(li);
    });
    sec.appendChild(ul);

    if (total > shown) {
      const overflow = document.createElement('div');
      overflow.className = 'command-overflow';
      overflow.textContent = `+ ${total - shown} more commands not shown`;
      sec.appendChild(overflow);
    }
    return sec;
  }

  function renderVaultFetchesSection(d) {
    const sec = section(`Vault fetches (${d.vault_fetches.length})`);
    const ul = document.createElement('ul');
    ul.className = 'vault-list';
    d.vault_fetches.forEach(v => {
      const li = document.createElement('li');
      const ts = document.createElement('span');
      ts.className = 'vault-ts';
      ts.textContent = formatTimestamp(v.ts);
      const usr = document.createElement('span');
      usr.className = 'vault-user';
      usr.textContent = v.user || '?';
      li.append(ts, usr);
      ul.appendChild(li);
    });
    sec.appendChild(ul);
    return sec;
  }

  // ---- Helpers ---------------------------------------------

  function describeLlmConfidence(score, call) {
    if (score === null || score === undefined) return 'no score';
    if (call === 'suspicious') {
      if (score >= 0.95) return 'very confident suspicious';
      if (score >= 0.8)  return 'confident suspicious';
      if (score >= 0.6)  return 'leaning suspicious';
      return 'borderline';
    }
    if (score <= 0.05) return 'very confident clean';
    if (score <= 0.2)  return 'confident clean';
    if (score <= 0.4)  return 'leaning clean';
    return 'borderline';
  }

  function formatTimestamp(iso) {
    if (!iso) return '—';
    return iso.replace('T', ' ').slice(0, 19);
  }

  // ----------------------------------------------------------
  //  Loader / error fallbacks
  // ----------------------------------------------------------

  function showLoadError(err) {
    document.body.classList.remove('is-loading');
    const tbody = document.getElementById('session-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 8;
    td.style.textAlign = 'center';
    td.style.padding = '3rem 0';
    td.style.color = 'var(--bad)';
    td.innerHTML = `Failed to load <code>data/summary.json</code> or <code>data/sessions_summary.json</code>.<br>
                    Run <code>python docs/build_dashboard.py</code> from <code>code/</code> to regenerate them.<br>
                    <small style="color:var(--fg-2)">${(err && err.message) || err}</small>`;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
})();
