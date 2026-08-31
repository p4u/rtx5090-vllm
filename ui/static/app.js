/* vllm-ui frontend — vanilla JS, no build step.
 * Polls /api/state (2s while launching, 10s idle) and /api/metrics + /api/gpu
 * (5s when a model is healthy). Tokens/s are client-side deltas of vLLM's
 * cumulative counters between polls. */

"use strict";

const $ = (id) => document.getElementById(id);

let stateTimer = null;
let metricsTimer = null;
let lastMetrics = null;      // previous counter sample for tokens/s deltas
let modelsCache = [];
let runningKey = null;
let launching = false;
let logSource = null;        // EventSource when following logs
let tokenVisible = false;
let tokenValue = null;
let apiBase = location.origin + "/v1";   // overridden by UI_DOMAIN via /api/state
let browsingAvailable = false;           // obscura image present (from /api/state)

// ─── tiny fetch helpers ────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (r.status === 401) { showLogin(); throw new Error("unauthorized"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

async function copyText(text) {
  // navigator.clipboard needs a secure context; this UI is often served over
  // plain HTTP on a LAN/VPN domain, so fall back to the legacy path.
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch {}
    ta.remove();
    return ok;
  }
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined) return "—";
  if (typeof n !== "number") return String(n);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(digits);
}

function fmtCtx(n) {
  if (!n) return "—";
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

// ─── auth ──────────────────────────────────────────────────────────────────

function showLogin() {
  stopTimers();
  $("main").hidden = true;
  $("login").hidden = false;
}

async function showMain() {
  $("login").hidden = true;
  $("main").hidden = false;
  $("api-url").textContent = apiBase;
  loadChats(); renderConvList(); renderMessages(); bindSettings();
  await Promise.all([loadModels(), loadTokenMasked(), loadBind()]);
  pollState();
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("login-error").hidden = true;
  try {
    await api("/auth/login", {
      method: "POST",
      body: JSON.stringify({ password: $("password").value }),
    });
    $("password").value = "";
    showMain();
  } catch {
    $("login-error").hidden = false;
  }
});

$("logout-btn").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST" });
  showLogin();
});

// ─── state polling ─────────────────────────────────────────────────────────

function stopTimers() {
  clearTimeout(stateTimer);
  clearTimeout(metricsTimer);
  if (logSource) { logSource.close(); logSource = null; }
}

async function pollState() {
  clearTimeout(stateTimer);
  let st;
  try { st = await api("/api/state"); }
  catch { stateTimer = setTimeout(pollState, 5000); return; }

  if (st.api_base && st.api_base !== apiBase) {
    apiBase = st.api_base;
    $("api-url").textContent = apiBase;
    renderToken();
  }
  browsingAvailable = !!st.browsing_available;
  renderBrowseToggle();
  runningKey = st.running_key;
  launching = st.launching;
  renderChatHeader();
  renderStatus(st);
  renderLaunch(st.launch);
  renderModels();       // running badge may have moved
  renderRuntime(st);

  if (st.healthy) scheduleMetrics();
  else updateVramStrip(null, null);
  stateTimer = setTimeout(pollState, launching ? 2000 : 10000);
}

function renderStatus(st) {
  const pill = $("status-pill");
  if (st.healthy && runningKey) {
    pill.textContent = `${runningKey} · healthy`;
    pill.className = "pill pill-ok";
  } else if (launching) {
    pill.textContent = `${(st.launch && st.launch.key) || runningKey || "model"} · starting`;
    pill.className = "pill pill-busy";
  } else if (runningKey) {
    const health = (st.container && (st.container.health || st.container.status)) || "?";
    pill.textContent = `${runningKey} · ${health}`;
    pill.className = health === "unhealthy" ? "pill pill-bad" : "pill pill-busy";
  } else {
    pill.textContent = "no model";
    pill.className = "pill pill-off";
  }
  $("stop-btn").hidden = !runningKey && !launching;
}

function renderLaunch(launch) {
  const banner = $("launch-banner");
  if (!launch || launch.phase === "ready") {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  banner.classList.toggle("failed", launch.phase === "failed");
  const pill = $("launch-phase");
  pill.textContent = launch.phase;
  pill.className = "pill " + (launch.phase === "failed" ? "pill-bad" : "pill-busy");
  $("launch-title").textContent = launch.key;
  $("launch-elapsed").textContent = launch.error
    ? launch.error : `${launch.elapsed}s elapsed`;
  const log = $("launch-log");
  log.textContent = (launch.log_tail || []).join("\n");
  log.scrollTop = log.scrollHeight;
}

$("stop-btn").addEventListener("click", async () => {
  if (!confirm("Stop the running model?")) return;
  $("stop-btn").disabled = true;
  try { await api("/api/stop", { method: "POST" }); }
  finally { $("stop-btn").disabled = false; }
  pollState();
});

// ─── models ────────────────────────────────────────────────────────────────

async function loadModels() {
  modelsCache = (await api("/api/models")).models;
  renderModels();
}

const OVERRIDE_FIELDS = [
  ["max_model_len", "Context length", "number"],
  ["gpu_memory_utilization", "GPU memory util", "number"],
  ["max_num_seqs", "Max concurrent seqs", "number"],
  ["max_num_batched_tokens", "Max batched tokens", "number"],
  ["kv_cache_dtype", "KV cache dtype", "select"],
];

function renderModels() {
  const grid = $("model-grid");
  const openDrawers = new Set(
    [...grid.querySelectorAll(".drawer:not([hidden])")].map((d) => d.dataset.key));
  grid.textContent = "";
  for (const m of modelsCache) {
    const card = document.createElement("div");
    card.className = "card model-card" + (m.key === runningKey ? " running" : "");

    const head = document.createElement("div");
    head.className = "model-head";
    const key = document.createElement("span");
    key.className = "model-key";
    key.textContent = m.key;
    head.appendChild(key);
    if (m.key === runningKey) {
      const pill = document.createElement("span");
      pill.className = "pill pill-ok";
      pill.textContent = "running";
      head.appendChild(pill);
    }
    card.appendChild(head);

    const name = document.createElement("div");
    name.className = "model-name";
    name.textContent = m.name;
    card.appendChild(name);

    const badges = document.createElement("div");
    badges.className = "badges";
    const hasOverrides = Object.values(m.overrides || {}).some((v) => v !== null && v !== "");
    for (const b of [
      `ctx ${fmtCtx(m.defaults.max_model_len || m.context_window)}`,
      m.reasoning ? "reasoning" : null,
      m.vision ? "vision" : null,
      hasOverrides ? "overridden" : null,
    ]) {
      if (!b) continue;
      const el = document.createElement("span");
      el.className = "badge";
      el.textContent = b;
      badges.appendChild(el);
    }
    card.appendChild(badges);

    const actions = document.createElement("div");
    actions.className = "model-actions";
    const startBtn = document.createElement("button");
    startBtn.className = "btn small-btn";
    startBtn.textContent = m.key === runningKey ? "Restart" : "Start";
    startBtn.disabled = launching;
    startBtn.addEventListener("click", () => startModel(m, collectOverrides(card, m)));
    const detailsBtn = document.createElement("button");
    detailsBtn.className = "btn ghost small-btn";
    detailsBtn.textContent = "Details";
    actions.append(startBtn, detailsBtn);
    card.appendChild(actions);

    const drawer = buildDrawer(m);
    drawer.hidden = !openDrawers.has(m.key);
    card.appendChild(drawer);
    detailsBtn.addEventListener("click", () => { drawer.hidden = !drawer.hidden; });

    grid.appendChild(card);
  }
}

function buildDrawer(m) {
  const drawer = document.createElement("div");
  drawer.className = "drawer";
  drawer.dataset.key = m.key;

  const note = document.createElement("p");
  note.className = "warn-note";
  note.textContent =
    "Defaults are empirically verified on a 32 GB 5090 — overrides may OOM. " +
    "After boot, check `GPU KV cache size` in the logs.";
  drawer.appendChild(note);

  for (const [field, label, kind] of OVERRIDE_FIELDS) {
    const row = document.createElement("div");
    row.className = "field";
    const lab = document.createElement("label");
    lab.textContent = label;
    let input;
    if (kind === "select") {
      input = document.createElement("select");
      for (const opt of ["", "auto", "fp8"]) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt === "" ? `default (${m.defaults[field] ?? "auto"})` : opt;
        input.appendChild(o);
      }
      input.value = m.overrides[field] ?? "";
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = field === "gpu_memory_utilization" ? "0.01" : "1";
      input.placeholder = `default: ${m.defaults[field] ?? "vLLM default"}`;
      input.value = m.overrides[field] ?? "";
    }
    input.dataset.field = field;
    row.append(lab, input);
    drawer.appendChild(row);
  }

  const extraRow = document.createElement("div");
  extraRow.className = "field";
  const extraLab = document.createElement("label");
  extraLab.textContent = "Extra vLLM args";
  const extraInput = document.createElement("input");
  extraInput.type = "text";
  extraInput.placeholder = "--enforce-eager --seed 0";
  extraInput.value = m.overrides.extra_args || "";
  extraInput.dataset.field = "extra_args";
  extraRow.append(extraLab, extraInput);
  drawer.appendChild(extraRow);

  const defaults = document.createElement("div");
  defaults.className = "kv small";
  const dl = document.createElement("div");
  const dlab = document.createElement("span");
  dlab.textContent = "verified launch flags";
  const dval = document.createElement("span");
  dval.textContent = (m.default_args || []).join(" ") || "—";
  dval.style.textAlign = "right";
  dl.append(dlab, dval);
  defaults.appendChild(dl);
  drawer.appendChild(defaults);

  const row = document.createElement("div");
  row.className = "model-actions";
  const resetBtn = document.createElement("button");
  resetBtn.className = "btn ghost small-btn";
  resetBtn.textContent = "Reset to verified defaults";
  resetBtn.addEventListener("click", () => {
    drawer.querySelectorAll("input, select").forEach((i) => { i.value = ""; });
  });
  row.appendChild(resetBtn);
  drawer.appendChild(row);
  return drawer;
}

function collectOverrides(card, m) {
  const ov = {};
  card.querySelectorAll(".drawer input, .drawer select").forEach((i) => {
    if (i.value !== "") ov[i.dataset.field] = i.value;
  });
  return ov;
}

async function startModel(m, overrides) {
  const switching = runningKey && runningKey !== m.key;
  if (switching &&
      !confirm(`Stop ${runningKey} and start ${m.key}? Reload takes 30–120s.`)) return;
  try {
    await api(`/api/models/${m.key}/start`, {
      method: "POST",
      body: JSON.stringify({ overrides }),
    });
  } catch (e) {
    alert(`Launch failed: ${e.message}`);
    return;
  }
  lastMetrics = null;
  launching = true;
  renderModels();
  pollState();
}

// ─── runtime dashboard ─────────────────────────────────────────────────────

function renderRuntime(st) {
  const has = !!st.running_key || launching;
  $("runtime-empty").hidden = has;
  $("runtime").hidden = !has;
  if (st.container) {
    const flags = st.container.effective_flags || {};
    const body = $("flags-body");
    body.textContent = "";
    for (const [k, v] of Object.entries(flags)) {
      const row = document.createElement("div");
      const kk = document.createElement("span");
      kk.textContent = k;
      const vv = document.createElement("span");
      vv.textContent = v === true ? "✓" : (Array.isArray(v) ? v.join(" ") : String(v));
      row.append(kk, vv);
      body.appendChild(row);
    }
  }
  if (!logSource) refreshLogsOnce();
}

function scheduleMetrics() {
  if (metricsTimer) return;
  const tick = async () => {
    metricsTimer = null;
    try {
      const [m, g] = await Promise.all([api("/api/metrics"), api("/api/gpu")]);
      renderMetrics(m);
      renderGpu(g);
    } catch {}
    if (runningKey) metricsTimer = setTimeout(tick, 5000);
  };
  tick();
}

function tile(label, value) {
  const t = document.createElement("div");
  t.className = "tile";
  const v = document.createElement("div");
  v.className = "v";
  v.textContent = value;
  const l = document.createElement("div");
  l.className = "l";
  l.textContent = label;
  t.append(v, l);
  return t;
}

function renderMetrics(m) {
  const tiles = $("tiles");
  tiles.textContent = "";
  if (!m.available) { tiles.appendChild(tile("metrics", "unavailable")); return; }

  // tokens/s from counter deltas between polls
  let genRate = null, promptRate = null;
  if (lastMetrics && m.ts > lastMetrics.ts) {
    const dt = m.ts - lastMetrics.ts;
    if (m.generation_tokens_total != null && lastMetrics.generation_tokens_total != null)
      genRate = (m.generation_tokens_total - lastMetrics.generation_tokens_total) / dt;
    if (m.prompt_tokens_total != null && lastMetrics.prompt_tokens_total != null)
      promptRate = (m.prompt_tokens_total - lastMetrics.prompt_tokens_total) / dt;
  }
  lastMetrics = m;

  tiles.appendChild(tile("gen tokens/s", genRate === null ? "—" : fmt(genRate)));
  tiles.appendChild(tile("prompt tokens/s", promptRate === null ? "—" : fmt(promptRate, 0)));
  tiles.appendChild(tile("KV cache", m.kv_cache_usage === null || m.kv_cache_usage === undefined
    ? "—" : `${fmt(m.kv_cache_usage * 100)}%`));
  tiles.appendChild(tile("running / waiting",
    `${fmt(m.requests_running, 0)} / ${fmt(m.requests_waiting, 0)}`));
  if (m.ttft) tiles.appendChild(tile("TTFT mean", `${fmt(m.ttft.mean * 1000, 0)} ms`));
  if (m.tpot) tiles.appendChild(tile("ms/token", fmt(m.tpot.mean * 1000)));
  if (m.e2e_latency) tiles.appendChild(tile("e2e p99",
    m.e2e_latency.p99 == null ? "—" : `${fmt(m.e2e_latency.p99)} s`));
  tiles.appendChild(tile("requests done", fmt(m.requests_success_total, 0)));
  if (m.spec_accept_rate !== undefined)
    tiles.appendChild(tile("spec accept", `${fmt(m.spec_accept_rate * 100, 0)}%`));
  if (m.preemptions_total)
    tiles.appendChild(tile("preemptions", fmt(m.preemptions_total, 0)));
}

function updateVramStrip(usedMib, totalMib) {
  const fill = $("vram-fill");
  const readout = $("vram-readout");
  if (usedMib == null || !totalMib) {
    fill.style.width = "0%";
    readout.textContent = "VRAM —";
    return;
  }
  const pct = Math.min(100, (usedMib / totalMib) * 100);
  fill.style.width = pct.toFixed(1) + "%";
  // ~96% is NORMAL for a max-context config (vLLM preallocates the KV pool);
  // red only past 98% — the allocator's true danger zone.
  fill.classList.toggle("hot", pct > 98);
  readout.textContent =
    `VRAM ${(usedMib / 1024).toFixed(1)}/${(totalMib / 1024).toFixed(0)} GiB`;
}

function renderGpu(g) {
  const body = $("gpu-body");
  body.textContent = "";
  updateVramStrip(g.available ? g.memory_used_mib : null,
                  g.available ? g.memory_total_mib : null);
  if (!g.available) {
    body.textContent = "No running container — GPU stats appear when a model is up.";
    return;
  }
  const rows = [
    ["VRAM", `${(g.memory_used_mib / 1024).toFixed(1)} / ${(g.memory_total_mib / 1024).toFixed(1)} GiB`],
    ["GPU util", `${g.utilization_pct}%`],
    ["Temperature", `${g.temperature_c} °C`],
    ["Power", `${g.power_w} / ${g.power_limit_w} W`],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement("div");
    const kk = document.createElement("span");
    kk.textContent = k;
    const vv = document.createElement("span");
    vv.textContent = v;
    row.append(kk, vv);
    body.appendChild(row);
  }
}

// ─── logs ──────────────────────────────────────────────────────────────────

async function refreshLogsOnce() {
  try {
    const r = await api("/api/logs?tail=100");
    const panel = $("log-panel");
    panel.textContent = r.logs;
    panel.scrollTop = panel.scrollHeight;
  } catch {}
}

$("follow-btn").addEventListener("click", () => {
  const btn = $("follow-btn");
  if (logSource) {
    logSource.close();
    logSource = null;
    btn.textContent = "Follow";
    return;
  }
  const panel = $("log-panel");
  panel.textContent = "";
  logSource = new EventSource("/api/logs/stream");
  logSource.onmessage = (e) => {
    panel.textContent += JSON.parse(e.data) + "\n";
    if (panel.textContent.length > 200000)
      panel.textContent = panel.textContent.slice(-100000);
    panel.scrollTop = panel.scrollHeight;
  };
  logSource.onerror = () => { logSource.close(); logSource = null; btn.textContent = "Follow"; };
  btn.textContent = "Stop following";
});

// ─── token panel ───────────────────────────────────────────────────────────

function renderToken() {
  $("token-value").textContent = tokenVisible && tokenValue
    ? tokenValue : "••••••••••••••••••••";
  $("token-show").textContent = tokenVisible ? "Hide" : "Show";
  $("curl-example").textContent =
    `curl ${apiBase}/chat/completions \\\n` +
    `  -H "Authorization: Bearer ${tokenVisible && tokenValue ? tokenValue : "<token>"}" \\\n` +
    `  -H "Content-Type: application/json" \\\n` +
    `  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'`;
}

async function loadTokenMasked() {
  tokenValue = (await api("/api/token")).token;
  renderToken();
}

// ─── serving bind ──────────────────────────────────────────────────────────

async function loadBind() {
  let b;
  try { b = await api("/api/bind"); } catch { return; }
  const sel = $("bind-host");
  sel.textContent = "";
  const envd = b.env_defaults || {};
  const envLabel = envd.HOST_IP ? `HOST_IP ${envd.HOST_IP}`
    : envd.BIND_CIDR ? `BIND_CIDR ${envd.BIND_CIDR}` : "0.0.0.0";
  const opts = [
    ["", `Default from .env (${envLabel})`],
    ["127.0.0.1", "127.0.0.1 — loopback only"],
    ...(b.interfaces || []).map((i) => [i.ip, `${i.ip} — ${i.ifname}`]),
    ["0.0.0.0", "0.0.0.0 — all interfaces"],
  ];
  for (const [value, label] of opts) {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  }
  sel.value = (b.current && b.current.host) || "";
  $("bind-port").value = (b.current && b.current.port) || "";
  $("bind-port").placeholder = `port (default ${envd.HOST_PORT || 8080})`;
}

$("bind-apply").addEventListener("click", async () => {
  const status = $("bind-status");
  status.textContent = "";
  try {
    await api("/api/bind", {
      method: "POST",
      body: JSON.stringify({
        host: $("bind-host").value || null,
        port: $("bind-port").value ? +$("bind-port").value : null,
      }),
    });
    status.textContent = "Saved — applies on the next model start.";
  } catch (e) {
    status.textContent = `Not saved: ${e.message}`;
  }
});

$("url-copy").addEventListener("click", async () => {
  await copyText(apiBase);
  $("url-copy").textContent = "Copied!";
  setTimeout(() => { $("url-copy").textContent = "Copy URL"; }, 1200);
});

$("token-show").addEventListener("click", () => {
  tokenVisible = !tokenVisible;
  renderToken();
});

$("token-copy").addEventListener("click", async () => {
  await copyText(tokenValue);
  $("token-copy").textContent = "Copied!";
  setTimeout(() => { $("token-copy").textContent = "Copy"; }, 1200);
});

$("token-renew").addEventListener("click", async () => {
  if (!confirm("Renew the API token? Every client using the old token stops working immediately. " +
               "The model's direct port keeps the old token until the next model start.")) return;
  tokenValue = (await api("/api/token/renew")).token;
  tokenVisible = true;
  renderToken();
});

// ─── tabs ──────────────────────────────────────────────────────────────────

function showTab(name) {
  for (const t of ["manage", "monitor", "chat"]) {
    $(`tab-${t}`).hidden = t !== name;
    $(`tab-btn-${t}`).classList.toggle("active", t === name);
  }
  if (name === "chat") { renderChatHeader(); $("chat-input").focus(); }
  if (name === "monitor") pollHistory();
  else stopHistory();
}
$("tab-btn-manage").addEventListener("click", () => showTab("manage"));
$("tab-btn-monitor").addEventListener("click", () => showTab("monitor"));
$("tab-btn-chat").addEventListener("click", () => showTab("chat"));

// ─── monitor: history charts ───────────────────────────────────────────────
// Hand-rolled SVG line charts (no chart library — the UI must work on an
// isolated VPN). Dataviz discipline: one series per chart (the title names
// it, no legend), thin 2px line, recessive grid, values in ink not series
// color, crosshair + tooltip on hover, gaps render as breaks.

let historyTimer = null;

const CHART_DEFS = [
  { key: "gpu_util", title: "GPU utilization", unit: "%", max: 100 },
  { key: "vram_used_gib", title: "VRAM used", unit: "GiB", maxKey: "vram_total_gib" },
  { key: "temp_c", title: "Temperature", unit: "°C" },
  { key: "power_w", title: "Power draw", unit: "W" },
  { key: "kv_pct", title: "KV cache usage", unit: "%", max: 100 },
  { key: "gen_rate", title: "Generation speed", unit: "tok/s" },
  { key: "prompt_rate", title: "Prefill speed", unit: "tok/s" },
  { key: "requests_running", title: "Requests running", unit: "" },
];

function deriveRates(samples) {
  // tokens/s from cumulative counter deltas; a counter drop = model restart.
  let prev = null;
  for (const s of samples) {
    s.gen_rate = null; s.prompt_rate = null;
    if (prev && s.ts > prev.ts) {
      const dt = s.ts - prev.ts;
      if (s.gen_tokens_total != null && prev.gen_tokens_total != null &&
          s.gen_tokens_total >= prev.gen_tokens_total)
        s.gen_rate = (s.gen_tokens_total - prev.gen_tokens_total) / dt;
      if (s.prompt_tokens_total != null && prev.prompt_tokens_total != null &&
          s.prompt_tokens_total >= prev.prompt_tokens_total)
        s.prompt_rate = (s.prompt_tokens_total - prev.prompt_tokens_total) / dt;
    }
    prev = s;
  }
}

const fmtClock = (ts) =>
  new Date(ts * 1000).toTimeString().slice(0, 5);

function chartTooltip() {
  let tip = document.getElementById("chart-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "chart-tip";
    tip.hidden = true;
    document.body.appendChild(tip);
  }
  return tip;
}

function drawChart(def, samples, interval) {
  const W = 600, H = 150, padL = 44, padR = 10, padT = 8, padB = 20;
  const card = document.createElement("div");
  card.className = "chart-card";

  const points = samples.map((s) => [s.ts, s[def.key]]);
  const vals = points.map((p) => p[1]).filter((v) => v != null);
  const latest = vals.length ? vals[vals.length - 1] : null;

  const head = document.createElement("div");
  head.className = "chart-head";
  const title = document.createElement("span");
  title.className = "chart-title";
  title.textContent = def.title;
  const cur = document.createElement("span");
  cur.className = "chart-cur";
  cur.textContent = latest === null ? "—" : `${fmt(latest)} ${def.unit}`;
  head.append(title, cur);
  card.appendChild(head);

  if (!vals.length) {
    const empty = document.createElement("div");
    empty.className = "muted chart-empty";
    empty.textContent = "no data yet";
    card.appendChild(empty);
    return card;
  }

  const t0 = points[0][0], t1 = points[points.length - 1][0];
  let yMax = def.max ?? Math.max(...vals) * 1.1;
  if (def.maxKey) {
    const totals = samples.map((s) => s[def.maxKey]).filter((v) => v != null);
    if (totals.length) yMax = totals[totals.length - 1];
  }
  if (yMax <= 0) yMax = 1;
  const x = (ts) => t1 === t0 ? padL : padL + (W - padL - padR) * (ts - t0) / (t1 - t0);
  const y = (v) => padT + (H - padT - padB) * (1 - Math.min(v, yMax) / yMax);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "chart-svg");
  const esc = (s) => String(s);
  let inner = "";
  // recessive grid: 3 horizontal lines + y labels in muted ink
  for (const frac of [0, 0.5, 1]) {
    const gy = padT + (H - padT - padB) * frac;
    const val = yMax * (1 - frac);
    inner += `<line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" class="grid"/>` +
             `<text x="${padL - 6}" y="${gy + 3.5}" class="tick" text-anchor="end">${fmt(val, val >= 100 ? 0 : 1)}</text>`;
  }
  inner += `<text x="${padL}" y="${H - 5}" class="tick">${fmtClock(t0)}</text>` +
           `<text x="${W - padR}" y="${H - 5}" class="tick" text-anchor="end">${fmtClock(t1)}</text>`;
  // series: break the line where samples are missing or far apart
  let seg = [];
  const segs = [];
  let prevTs = null;
  for (const [ts, v] of points) {
    const gap = prevTs !== null && ts - prevTs > interval * 3;
    if (v == null || gap) { if (seg.length) segs.push(seg); seg = []; }
    if (v != null) seg.push(`${x(ts).toFixed(1)},${y(v).toFixed(1)}`);
    prevTs = ts;
  }
  if (seg.length) segs.push(seg);
  for (const sg of segs) {
    if (sg.length === 1) {
      const [px, py] = sg[0].split(",");
      inner += `<circle cx="${px}" cy="${py}" r="2.5" class="series-dot"/>`;
    } else {
      inner += `<polyline points="${sg.join(" ")}" class="series"/>`;
    }
  }
  inner += `<line class="crosshair" x1="0" y1="${padT}" x2="0" y2="${H - padB}" visibility="hidden"/>` +
           `<circle class="hover-dot" r="3.5" visibility="hidden"/>`;
  svg.innerHTML = inner;
  card.appendChild(svg);

  // hover layer: nearest-sample crosshair + tooltip
  const tip = chartTooltip();
  const cross = svg.querySelector(".crosshair");
  const hdot = svg.querySelector(".hover-dot");
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const ts = t0 + (t1 - t0) * Math.min(Math.max(((e.clientX - r.left) / r.width * W - padL) / (W - padL - padR), 0), 1);
    let best = null;
    for (const [pts, v] of points)
      if (v != null && (best === null || Math.abs(pts - ts) < Math.abs(best[0] - ts))) best = [pts, v];
    if (!best) return;
    cross.setAttribute("x1", x(best[0])); cross.setAttribute("x2", x(best[0]));
    cross.removeAttribute("visibility");
    hdot.setAttribute("cx", x(best[0])); hdot.setAttribute("cy", y(best[1]));
    hdot.removeAttribute("visibility");
    tip.hidden = false;
    tip.textContent = `${fmt(best[1])} ${def.unit} · ${fmtClock(best[0])}`;
    tip.style.left = `${e.clientX + 12}px`;
    tip.style.top = `${e.clientY - 28}px`;
  });
  svg.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    hdot.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  return card;
}

async function pollHistory() {
  clearTimeout(historyTimer);
  let h;
  try { h = await api("/api/history"); }
  catch { historyTimer = setTimeout(pollHistory, 10000); return; }
  const samples = h.samples || [];
  deriveRates(samples);
  $("charts-empty").hidden = samples.length > 0;
  const grid = $("charts");
  grid.textContent = "";
  if (samples.length) {
    for (const def of CHART_DEFS) grid.appendChild(drawChart(def, samples, h.interval));
  }
  if (!$("tab-monitor").hidden) historyTimer = setTimeout(pollHistory, 10000);
}

function stopHistory() {
  clearTimeout(historyTimer);
  historyTimer = null;
}

// ─── chat: minimal markdown renderer ───────────────────────────────────────

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");
}

function renderMarkdown(src) {
  // NUL delimits our code-block placeholders below and is never legitimate
  // chat text — strip it so crafted input can't forge a placeholder.
  src = src.replace(/\u0000/g, "");
  // 1. pull fenced code blocks out so nothing inside them is transformed.
  // Line-based scan: a fence only opens/closes at the start of a line, so
  // adjacent or nested fences in model output (```markdown containing ```py)
  // can't bleed code into the prose the way a lazy regex does.
  const codes = [];
  const kept = [];
  let fence = null;   // {lang, lines} while inside a block
  for (const line of src.split("\n")) {
    const open = fence === null && line.match(/^\s*```(\S*)\s*$/);
    if (open) {
      fence = { lang: open[1], lines: [] };
    } else if (fence !== null && /^\s*```\s*$/.test(line)) {
      codes.push({ lang: fence.lang, code: fence.lines.join("\n") });
      kept.push(`\u0000CODE${codes.length - 1}\u0000`);
      fence = null;
    } else if (fence !== null) {
      fence.lines.push(line);
    } else {
      kept.push(line);
    }
  }
  // Unterminated fence (mid-stream, or a stray trailing ``` from the model):
  // render its content as code, but drop it entirely while still empty so a
  // lone closing fence never leaves an empty box behind.
  if (fence !== null && fence.lines.length) {
    codes.push({ lang: fence.lang, code: fence.lines.join("\n") });
    kept.push(`\u0000CODE${codes.length - 1}\u0000`);
  }
  let html = escapeHtml(kept.join("\n"));

  // tables (| a | b | with a |---| separator row)
  html = html.replace(
    /(^\|.+\|\n\|[\s\-:|]+\|\n(?:\|.+\|\n?)*)/gm,
    (block) => {
      const rows = block.trim().split("\n").map((r) =>
        r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
      const head = rows.shift(); rows.shift(); // header + separator
      const tr = (cells, tag) =>
        `<tr>${cells.map((c) => `<${tag}>${c}</${tag}>`).join("")}</tr>`;
      return `<table>${tr(head, "th")}${rows.map((r) => tr(r, "td")).join("")}</table>`;
    });

  html = html
    .replace(/^###+ (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/^&gt; ?(.*)$/gm, "<blockquote>$1</blockquote>")
    .replace(/^(?:-{3,}|\*{3,})$/gm, "<hr>")
    .replace(/^[*\-] (.*)$/gm, "<li>$1</li>")
    .replace(/^\d+\. (.*)$/gm, "<li data-ol>$1</li>")
    .replace(/(<li data-ol>[\s\S]*?<\/li>)(?!\n<li data-ol)/g, "<ol>$1</ol>")
    .replace(/(<li>(?:(?!<li data-ol)[\s\S])*?<\/li>)(?!\n<li>)/g, "<ul>$1</ul>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>");
  html = `<p>${html}</p>`;

  return html.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => {
    const { lang, code } = codes[+i];
    return `<pre><button class="btn ghost small-btn copy-code">Copy</button>` +
           `<code data-lang="${escapeHtml(lang)}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
  });
}

document.addEventListener("click", (e) => {
  if (e.target.classList && e.target.classList.contains("copy-code")) {
    copyText(e.target.nextElementSibling.textContent);
    e.target.textContent = "Copied!";
    setTimeout(() => { e.target.textContent = "Copy"; }, 1200);
  }
});

// ─── chat: conversation store (localStorage) ───────────────────────────────

const CHATS_KEY = "vllm-ui-chats";
const SETTINGS_KEY = "vllm-ui-chat-settings";
let chats = [];
let activeChatId = null;
let generating = false;
let genAbort = null;
let pendingFiles = [];   // [{name, kind:'text'|'image', content?, dataurl?, size}]

function loadChats() {
  try { chats = JSON.parse(localStorage.getItem(CHATS_KEY)) || []; }
  catch { chats = []; }
  if (chats.length) activeChatId = chats[0].id;
}

function saveChats() {
  try { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); }
  catch { /* quota (large images) — chat keeps working, history just won't persist */ }
}

function activeChat() { return chats.find((c) => c.id === activeChatId) || null; }

function newChat() {
  const c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
              title: "New chat", created: Date.now(), messages: [] };
  chats.unshift(c);
  activeChatId = c.id;
  saveChats(); renderConvList(); renderMessages();
}

function deleteChat(id) {
  chats = chats.filter((c) => c.id !== id);
  if (activeChatId === id) activeChatId = chats.length ? chats[0].id : null;
  saveChats(); renderConvList(); renderMessages();
}

function chatMatches(c, q) {
  if (!q) return true;
  q = q.toLowerCase();
  return c.title.toLowerCase().includes(q) ||
         c.messages.some((m) => (m.text || "").toLowerCase().includes(q));
}

function renderConvList() {
  const list = $("conv-list");
  const q = $("chat-search").value.trim();
  list.textContent = "";
  for (const c of chats) {
    if (!chatMatches(c, q)) continue;
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === activeChatId ? " active" : "");
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = c.title;
    title.title = "Double-click to rename";
    title.addEventListener("dblclick", (e) => { e.stopPropagation(); startRename(c, item, title); });

    const rename = iconBtn("✎", "Rename chat", (e) => {
      e.stopPropagation(); startRename(c, item, title);
    });
    const exp = iconBtn("⤓", "Export as Markdown", (e) => {
      e.stopPropagation(); exportChatMarkdown(c);
    });
    const del = iconBtn("✕", "Delete chat", (e) => {
      e.stopPropagation();
      if (confirm(`Delete chat "${c.title}"?`)) deleteChat(c.id);
    });
    del.classList.add("danger-icon");
    item.append(title, rename, exp, del);
    item.addEventListener("click", () => {
      activeChatId = c.id;
      renderConvList(); renderMessages();
    });
    list.appendChild(item);
  }
}

function iconBtn(glyph, tip, onClick) {
  const b = document.createElement("button");
  b.className = "del icon-btn";
  b.textContent = glyph;
  b.title = tip;
  b.addEventListener("click", onClick);
  return b;
}

function startRename(c, item, titleEl) {
  const input = document.createElement("input");
  input.className = "rename-input";
  input.value = c.title;
  input.maxLength = 80;
  item.replaceChild(input, titleEl);
  input.focus();
  input.select();
  const commit = () => {
    const v = input.value.trim();
    if (v) { c.title = v; c.renamed = true; saveChats(); }
    renderConvList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") renderConvList();
  });
  input.addEventListener("blur", commit);
  input.addEventListener("click", (e) => e.stopPropagation());
}

$("chat-search").addEventListener("input", renderConvList);

// ─── chat: markdown export ─────────────────────────────────────────────────

function chatToMarkdown(c) {
  const lines = [`# ${c.title}`, "",
    `> Exported from vllm-ui on ${new Date().toLocaleString()} · ` +
    `${c.messages.length} messages`, ""];
  for (const m of c.messages) {
    const when = m.ts ? ` — ${new Date(m.ts).toLocaleString()}` : "";
    if (m.role === "user") {
      lines.push(`## You${when}`, "");
      for (const f of m.files || [])
        lines.push(`*Attached: ${f.name}*`, "");
      lines.push(m.text || "", "");
    } else {
      lines.push(`## ${m.model || "Assistant"}${when}`, "");
      for (const b of m.browsing || []) {
        if (b.event === "tool_start")
          lines.push(b.tool === "web_search"
            ? `*Searched the web: ${b.args.query ?? ""}*`
            : `*Fetched: ${b.args.url ?? ""}*`);
      }
      if ((m.browsing || []).length) lines.push("");
      if (m.reasoning)
        lines.push("<details><summary>Reasoning</summary>", "",
                   m.reasoning.trim(), "", "</details>", "");
      lines.push(m.text || "", "");
      if (m.stats && m.stats.tokens)
        lines.push(`*${m.stats.tokens} tokens · ${m.stats.tps} tok/s*`, "");
    }
    lines.push("---", "");
  }
  return lines.join("\n");
}

function exportChatMarkdown(c) {
  const blob = new Blob([chatToMarkdown(c)], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = c.title.replace(/[^\w\d àèéíòóúç.-]+/gi, "_").slice(0, 60) + ".md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ─── chat: message rendering ───────────────────────────────────────────────

function renderChatHeader() {
  const el = $("chat-model-name");
  if (runningKey) {
    const m = modelsCache.find((x) => x.key === runningKey);
    el.textContent = `Chatting with ${runningKey}`;
    el.nextElementSibling.textContent = m && m.vision
      ? "This model accepts images — attach files or pictures below."
      : "Text-only model — attached text files are inlined into your message.";
  } else {
    el.textContent = "No model running";
    el.nextElementSibling.textContent =
      "Start one from the Manage tab, then come back here.";
  }
}

const fmtTime = (ts) => new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function msgEl(role, streaming = false) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const who = document.createElement("div");
  who.className = "who";
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (streaming ? " streaming" : "");
  wrap.append(who, bubble);
  return wrap;
}

function renderWho(wrap, m) {
  const who = wrap.querySelector(".who");
  who.textContent = "";
  const name = document.createElement("span");
  name.textContent = m.role === "user" ? "You" : (m.model || runningKey || "model");
  who.appendChild(name);
  if (m.ts) {
    const t = document.createElement("span");
    t.className = "msg-meta";
    t.textContent = fmtTime(m.ts);
    who.appendChild(t);
  }
  if (m.role !== "user" && m.stats && m.stats.tokens) {
    const s = document.createElement("span");
    s.className = "msg-meta";
    s.textContent = `${m.stats.tokens} tok · ${m.stats.tps} tok/s`;
    who.appendChild(s);
  }
}

function msgActions(wrap, m, chat, idx) {
  const row = document.createElement("div");
  row.className = "msg-actions";
  const copy = document.createElement("button");
  copy.className = "btn ghost small-btn";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    await copyText(m.text || "");
    copy.textContent = "Copied!";
    setTimeout(() => { copy.textContent = "Copy"; }, 1200);
  });
  row.appendChild(copy);
  if (m.role === "user" && !generating) {
    const edit = document.createElement("button");
    edit.className = "btn ghost small-btn";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => startEditMessage(wrap, m, chat, idx));
    row.appendChild(edit);
  }
  // regenerate: only on the final assistant message
  if (m.role === "assistant" && idx === chat.messages.length - 1 && !generating) {
    const regen = document.createElement("button");
    regen.className = "btn ghost small-btn";
    regen.textContent = "Regenerate";
    regen.addEventListener("click", async () => {
      if (generating) return;
      chat.messages.pop();
      renderMessages();
      await generate(chat);
    });
    row.appendChild(regen);
  }
  return row;
}

function startEditMessage(wrap, m, chat, idx) {
  // AnythingLLM-style edit-and-resubmit: saving truncates everything below
  // this message and regenerates from here.
  const bubble = wrap.querySelector(".bubble");
  bubble.textContent = "";
  const ta = document.createElement("textarea");
  ta.className = "edit-textarea";
  ta.value = m.text;
  const actions = document.createElement("div");
  actions.className = "model-actions";
  const save = document.createElement("button");
  save.className = "btn small-btn";
  save.textContent = "Save & resubmit";
  save.addEventListener("click", async () => {
    const v = ta.value.trim();
    if (!v) return;
    if (idx < chat.messages.length - 1 &&
        !confirm("Saving resubmits from here — later messages in this chat are discarded.")) return;
    m.text = v;
    chat.messages.length = idx + 1;   // truncate below
    saveChats(); renderMessages();
    await generate(chat);
  });
  const cancel = document.createElement("button");
  cancel.className = "btn ghost small-btn";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", renderMessages);
  actions.append(save, cancel);
  bubble.append(ta, actions);
  ta.focus();
  ta.style.height = Math.min(ta.scrollHeight + 4, 300) + "px";
}

function renderMessageInto(wrap, m) {
  const bubble = wrap.querySelector(".bubble");
  if (m.role === "user") {
    bubble.textContent = m.text;
    if (m.files && m.files.length) {
      const files = document.createElement("div");
      files.className = "msg-attachments";
      for (const f of m.files) {
        const chip = document.createElement("span");
        chip.className = "attach-chip";
        if (f.kind === "image" && f.dataurl) {
          const img = document.createElement("img");
          img.src = f.dataurl; img.alt = f.name;
          chip.appendChild(img);
        }
        const nm = document.createElement("span");
        nm.textContent = f.name;
        chip.appendChild(nm);
        files.appendChild(chip);
      }
      wrap.insertBefore(files, bubble);
    }
  } else {
    bubble.textContent = "";
    if (m.browsing && m.browsing.length) {
      bubble.appendChild(renderBrowsing(wrap, m));
    }
    if (m.reasoning) {
      const det = document.createElement("details");
      det.className = "reasoning";
      const sum = document.createElement("summary");
      sum.textContent = "Reasoning";
      const body = document.createElement("div");
      body.className = "r-body";
      body.textContent = m.reasoning;
      det.append(sum, body);
      bubble.appendChild(det);
    }
    const content = document.createElement("div");
    content.innerHTML = renderMarkdown(m.text || "");
    bubble.appendChild(content);
    if (m.error) {
      const err = document.createElement("div");
      err.className = "error-bubble";
      err.textContent = m.error;
      bubble.appendChild(err);
    }
  }
}

function renderMessages() {
  const box = $("chat-messages");
  box.textContent = "";
  const chat = activeChat();
  $("chat-empty").hidden = !!(chat && chat.messages.length);
  if (!chat) return;
  chat.messages.forEach((m, idx) => {
    const el = msgEl(m.role);
    renderMessageInto(el, m);
    renderWho(el, m);
    el.appendChild(msgActions(el, m, chat, idx));
    box.appendChild(el);
  });
  box.scrollTop = box.scrollHeight;
}

// ─── chat: attachments ─────────────────────────────────────────────────────

const TEXT_MAX = 512 * 1024;      // 512 KB per text file
const IMAGE_MAX = 8 * 1024 * 1024; // 8 MB per image

$("attach-btn").addEventListener("click", (e) => { e.preventDefault(); $("file-input").click(); });

$("file-input").addEventListener("change", async () => {
  for (const file of $("file-input").files) {
    if (file.type.startsWith("image/")) {
      if (file.size > IMAGE_MAX) { alert(`${file.name}: image too large (max 8 MB)`); continue; }
      const dataurl = await new Promise((res) => {
        const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file);
      });
      pendingFiles.push({ name: file.name, kind: "image", dataurl, size: file.size });
    } else {
      if (file.size > TEXT_MAX) { alert(`${file.name}: file too large (max 512 KB)`); continue; }
      const content = await file.text();
      if (content.includes("\u0000")) { alert(`${file.name}: binary files are not supported`); continue; }
      pendingFiles.push({ name: file.name, kind: "text", content, size: file.size });
    }
  }
  $("file-input").value = "";
  renderAttachList();
});

function renderAttachList() {
  const list = $("attach-list");
  list.textContent = "";
  pendingFiles.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "attach-chip";
    if (f.kind === "image") {
      const img = document.createElement("img");
      img.src = f.dataurl; img.alt = f.name;
      chip.appendChild(img);
    }
    const nm = document.createElement("span");
    nm.textContent = `${f.name} (${(f.size / 1024).toFixed(0)} KB)`;
    const x = document.createElement("button");
    x.className = "x"; x.textContent = "✕";
    x.addEventListener("click", () => { pendingFiles.splice(i, 1); renderAttachList(); });
    chip.append(nm, x);
    list.appendChild(chip);
  });
}

// ─── chat: sending / streaming ─────────────────────────────────────────────

function chatSettings() {
  let s = {};
  try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; } catch {}
  return s;
}
function bindSettings() {
  const s = chatSettings();
  $("set-system").value = s.system || "";
  $("set-temperature").value = s.temperature ?? "";
  $("set-max-tokens").value = s.max_tokens ?? "";
  for (const id of ["set-system", "set-temperature", "set-max-tokens"]) {
    $(id).addEventListener("change", () => {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        system: $("set-system").value || undefined,
        temperature: $("set-temperature").value ? +$("set-temperature").value : undefined,
        max_tokens: $("set-max-tokens").value ? +$("set-max-tokens").value : undefined,
      }));
    });
  }
}

function apiMessages(chat) {
  // Rebuild the OpenAI messages array from stored history.
  const s = chatSettings();
  const out = [];
  if (s.system) out.push({ role: "system", content: s.system });
  for (const m of chat.messages) {
    if (m.error && !m.text) continue;            // skip failed empty replies
    if (m.role === "assistant") {
      out.push({ role: "assistant", content: m.text });
      continue;
    }
    let text = m.text;
    const images = [];
    for (const f of m.files || []) {
      if (f.kind === "text")
        text += `\n\n\`\`\`${f.name}\n${f.content}\n\`\`\``;
      else if (f.dataurl)
        images.push({ type: "image_url", image_url: { url: f.dataurl } });
    }
    out.push(images.length
      ? { role: "user", content: [{ type: "text", text }, ...images] }
      : { role: "user", content: text });
  }
  return out;
}

async function sendMessage() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (generating || (!text && !pendingFiles.length)) return;
  if (!runningKey) { alert("No model is running — start one from the Manage tab."); return; }
  if (!activeChat()) newChat();
  const chat = activeChat();

  const userMsg = { role: "user", text, files: pendingFiles, ts: Date.now() };
  chat.messages.push(userMsg);
  if (chat.title === "New chat" && !chat.renamed && text)
    chat.title = text.slice(0, 42) + (text.length > 42 ? "…" : "");
  pendingFiles = [];
  input.value = ""; input.style.height = "auto";
  renderAttachList(); renderConvList(); renderMessages();
  await generate(chat);
}

async function generate(chat) {
  const s = chatSettings();
  const assistantMsg = { role: "assistant", text: "", reasoning: "",
                         ts: Date.now(), model: runningKey };
  chat.messages.push(assistantMsg);
  const el = msgEl("assistant", true);
  renderWho(el, assistantMsg);
  $("chat-messages").appendChild(el);
  $("chat-empty").hidden = true;
  let usageTokens = 0;      // accumulated across browsing rounds
  let firstDeltaAt = 0;

  generating = true;
  genAbort = new AbortController();
  $("send-btn").hidden = true;
  $("stop-gen-btn").hidden = false;

  const box = $("chat-messages");
  const nearBottom = () => box.scrollHeight - box.scrollTop - box.clientHeight < 120;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: genAbort.signal,
      body: JSON.stringify({
        model: "default",
        stream: true,
        browsing: browsingEnabled(),
        // history minus the empty assistant placeholder just appended
        messages: apiMessages({ ...chat, messages: chat.messages.slice(0, -1) }),
        max_tokens: s.max_tokens || 8192,
        stream_options: { include_usage: true },
        ...(s.temperature !== undefined ? { temperature: s.temperature } : {}),
      }),
    });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { msg = (await resp.json()).error.message; } catch {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let lastDraw = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      let browsingChanged = false;
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(data); } catch { continue; }
        if (obj.browsing) {
          (assistantMsg.browsing ??= []).push(obj.browsing);
          browsingChanged = true;
          continue;
        }
        if (obj.error && obj.error.message) {
          assistantMsg.error = obj.error.message;
          continue;
        }
        if (obj.usage && obj.usage.completion_tokens)
          usageTokens += obj.usage.completion_tokens;
        const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
        if (!delta) continue;
        if (!firstDeltaAt) firstDeltaAt = performance.now();
        if (delta.content) assistantMsg.text += delta.content;
        // field renamed across vLLM versions — accept both
        const r = delta.reasoning_content ?? delta.reasoning;
        if (r) assistantMsg.reasoning += r;
      }
      if (browsingChanged) {   // activity rows render immediately, unthrottled
        const stick = nearBottom();
        renderMessageInto(el, assistantMsg);
        if (stick) box.scrollTop = box.scrollHeight;
        lastDraw = performance.now();
      }
      const now = performance.now();
      if (now - lastDraw > 80) {   // throttle re-renders during fast streams
        lastDraw = now;
        const stick = nearBottom();
        renderMessageInto(el, assistantMsg);
        if (stick) box.scrollTop = box.scrollHeight;
      }
    }
  } catch (e) {
    if (e.name !== "AbortError")
      assistantMsg.error = `Generation failed: ${e.message}`;
  } finally {
    generating = false;
    genAbort = null;
    $("send-btn").hidden = false;
    $("stop-gen-btn").hidden = true;
    if (usageTokens && firstDeltaAt) {
      const dur = (performance.now() - firstDeltaAt) / 1000;
      assistantMsg.stats = { tokens: usageTokens,
                             tps: +(usageTokens / Math.max(dur, 0.1)).toFixed(1) };
    }
    saveChats();
    renderMessages();   // full re-render attaches meta + action buttons
  }
}

// ─── web browsing toggle ───────────────────────────────────────────────────

function browsingEnabled() {
  return browsingAvailable && chatSettings().browsing !== false;
}

function renderBrowseToggle() {
  const btn = $("browse-toggle");
  btn.disabled = !browsingAvailable;
  btn.title = browsingAvailable
    ? "Let the model search and read the web (obscura)"
    : "Web browsing unavailable — obscura image not pulled (see run-ui.sh)";
  btn.setAttribute("aria-pressed", String(browsingEnabled()));
  btn.classList.toggle("toggled-on", browsingEnabled());
}

$("browse-toggle").addEventListener("click", () => {
  const s = chatSettings();
  s.browsing = !(s.browsing !== false);
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  renderBrowseToggle();
});

function renderBrowsing(wrap, m) {
  // activity block above the assistant content: one row per tool call,
  // spinner while a start has no matching result, expandable debug logs.
  // Everything is set via textContent — fetched data never becomes HTML.
  const box = document.createElement("div");
  box.className = "browse-activity";
  const events = m.browsing || [];
  const starts = events.filter((e) => e.event === "tool_start");
  const results = events.filter((e) => e.event === "tool_result");
  starts.forEach((e, i) => {
    const row = document.createElement("div");
    row.className = "browse-row";
    const what = document.createElement("span");
    what.className = "browse-what";
    what.textContent = e.tool === "web_search"
      ? `Searched: ${e.args.query ?? ""}`
      : `Fetched: ${e.args.url ?? ""}`;
    row.appendChild(what);
    const res = results[i];
    const status = document.createElement("span");
    status.className = "browse-status";
    if (!res) {
      status.textContent = "…";
      status.classList.add("busy");
    } else {
      status.textContent = res.ok
        ? `${res.preview}${res.duration ? ` · ${res.duration}s` : ""}`
        : `failed: ${res.preview}`;
      if (!res.ok) status.classList.add("bad");
    }
    row.appendChild(status);
    box.appendChild(row);
  });
  for (const e of events.filter((ev) => ev.event === "error")) {
    const row = document.createElement("div");
    row.className = "browse-row browse-error";
    row.textContent = e.message;
    box.appendChild(row);
  }
  const debugLines = results.flatMap((r) => r.debug || []);
  if (debugLines.length) {
    const det = document.createElement("details");
    det.className = "browse-debug";
    const sum = document.createElement("summary");
    sum.textContent = `Browsing debug log (${debugLines.length} lines)`;
    const pre = document.createElement("pre");
    pre.className = "log small";
    pre.textContent = debugLines.join("\n");
    det.append(sum, pre);
    box.appendChild(det);
  }
  return box;
}

$("send-btn").addEventListener("click", sendMessage);
$("stop-gen-btn").addEventListener("click", () => genAbort && genAbort.abort());
$("new-chat-btn").addEventListener("click", newChat);

$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("chat-input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
});

// ─── boot ──────────────────────────────────────────────────────────────────

(async () => {
  try {
    await api("/api/state");   // session check
    showMain();
  } catch {
    showLogin();
  }
})();
