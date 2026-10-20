/* Seller vs Holder dashboard: state, animation loop and the card renderers. Needs Engine and DATA. */
(function () {
  "use strict";
  const E = Engine.create(DATA);
  const N = E.N, MK = DATA.market;
  const SETS = DATA.sets || { "0.9": { alpha: 0.9, h_keys: DATA.h_keys, s_keys: DATA.s_keys, matrix: DATA.matrix, hists: DATA.hists, logs: DATA.logs, duals: DATA.duals, iters: { gda: 1500, lsm: 1500 } } };
  const LEVELS = Object.keys(SETS).sort((a, b) => +a - +b);
  const DAYS = MK.T * 252;                       // 126 trading days
  const dayOf = (n) => Math.round(n * DAYS / N);
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;
  const tok = (n) => getComputedStyle(root).getPropertyValue(n).trim();
  const C = {};
  for (const k of ["ink", "ink-2", "ink-3", "grid", "line", "panel", "panel-2", "seller", "holder", "third", "liq-lo", "liq-hi", "bg"]) C[k] = tok("--" + k);
  const FONT = (px, w = 400) => `${w} ${px}px "IBM Plex Sans", system-ui, sans-serif`;
  const MONO = (px) => `400 ${px}px "IBM Plex Mono", ui-monospace, monospace`;
  const money = (v, d = 2) => (v < -0.5 * 10 ** -d ? "−" : "") + "$" + Math.abs(v).toFixed(d);
  const num = (v, d = 2) => (v < -0.5 * 10 ** -d ? "−" : "") + Math.abs(v).toFixed(d);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const HEDGES = [
    { k: "gda", name: "GDA hedge", desc: "Trained alongside its own attacker by gradient descent–ascent, so it minimises the worst tail loss an exercise rule can cause." },
    { k: "lsm", name: "LSM-trained", desc: "Trained assuming the holder follows the Longstaff–Schwartz exercise rule. It never saw an attacker." },
    { k: "mat", name: "Maturity-trained", desc: "Trained assuming the holder never exercises early." },
    { k: "delta", name: "Merton delta", desc: "The textbook hedge: the European put delta under Merton jumps. It ignores trading costs and illiquidity." },
    { k: "none", name: "No hedge", desc: "Sells the put and holds no shares." },
  ];
  const STOPS = [
    { k: "advA", name: "Attacker vs GDA", role: "Worst case for the seller", desc: "Trained alongside the GDA hedge to maximise the seller's tail loss. It sees the seller's position and running loss." },
    { k: "advB", name: "Attacker vs LSM-trained", role: "Worst case for the seller", desc: "Trained to attack the LSM-trained hedge, which stayed fixed while it learned. It sees the seller's position and running loss." },
    { k: "advC", name: "Attacker vs maturity", role: "Worst case for the seller", desc: "Trained to attack the maturity-trained hedge, which stayed fixed while it learned. It sees the seller's position and running loss." },
    { k: "lsm_rule", name: "Rational holder", role: "Maximises own value", desc: "Exercises when the payoff beats the Longstaff–Schwartz continuation value. Ignores the seller entirely." },
    { k: "never", name: "Never early", role: "Holds to expiry", desc: "Exercises only at maturity, and only if in the money." },
  ];
  const SETTLES = [
    { k: "cash_lam1", name: "Cash", desc: "Cash settled: the holder receives K − S. No trading needed." },
    { k: "phys_lam1", name: "Physical · small", desc: "Physically settled, small holder (5% of daily volume): exercising means buying the share at today's spread and impact." },
    { k: "phys_psi1", name: "Physical · large", desc: "Physically settled, large holder (a full day's volume): impact is about 4.5 times larger." },
  ];
  const nameOf = (list, k) => list.find((x) => x.k === k);

  // ---------------- state ----------------
  const S = {
    mode: "game", hedge: "gda", stop: "advA", settle: "phys_lam1", alpha: "0.9",
    playing: !reduced, speed: 1, n: 0, settled: false, settleT: 0, rounds: 0,
    rng: Engine.makeRng(20261016), path: null, game: null, ref: null, hold: null,
    live: {}, hlive: {}, hex: {}, hprop: {}, duel: {}, shadow: null, batching: false, hover: {}, trainT: 0, dirty: true,
  };
  const SET = () => SETS[S.alpha];
  const aNum = () => SET().alpha;
  const pctTail = () => Math.round((1 - aNum()) * 100);             // the worst x% that CVaR averages
  const cvarName = () => (aNum() === 0 ? "Mean loss" : `CVaR ${Math.round(aNum() * 100)}%`);
  // trained models at levels other than 0.9 are stored as "gda@0.95", "advA@0.95", ...
  const kindOf = (base) => (S.alpha === "0.9" || !["gda", "lsm", "mat", "advA", "advB", "advC"].includes(base) ? base : `${base}@${S.alpha}`);
  const play = (h, st, p) => E.playRound(kindOf(h), kindOf(st), p);
  const mkey = () => `${S.hedge}|${S.stop}`;                          // key into this level's payoff matrix
  const key = () => `${S.alpha}|${S.hedge}|${S.stop}`;                // key for rounds played here
  // the comparison hedge played silently on every path: LSM-trained, or GDA when the seller is already LSM-trained
  const shadowKind = () => (S.hedge === "lsm" ? "gda" : "lsm");
  const duelKey = () => `${S.alpha}|${S.hedge}|${shadowKind()}|${S.stop}`;

  function compute() {
    S.game = play(S.hedge, S.stop, S.path);
    S.shadow = play(shadowKind(), S.stop, S.path);
    S.ref = E.rollout("delta", S.path);
    S.hold = E.holderRound(S.settle, S.path);
  }
  function newRound() {
    S.path = E.simulate(S.rng);
    compute();
    S.n = 0; S.settled = false; S.dirty = true;
  }
  const tau = () => (S.mode === "holder" ? S.hold.tau : S.game.tau);
  const ELL_EDGES = [1.25, 2, 4];
  const ellBucket = (ell) => ELL_EDGES.filter((e) => ell >= e).length;
  function recordGame(g, sh) {
    (S.live[key()] ||= []).push(g.loss);
    if (!sh) return;
    (S.live[`${S.alpha}|${shadowKind()}|${S.stop}`] ||= []).push(sh.loss);
    (S.duel[duelKey()] ||= []).push([g.loss, sh.loss, g.tau, sh.tau]);
  }
  function recordHolder(h, p, settle = S.settle) {
    (S.hlive[settle] ||= []).push(h.payoff);
    const pr = (S.hprop[settle] ||= { alive: [0, 0, 0, 0], ex: [0, 0, 0, 0] });
    for (let n = 1; n <= Math.min(h.tau, N - 1); n++) if (p.S[n] < 92) {
      const b = ellBucket(p.ell[n]); pr.alive[b]++; if (n === h.tau) pr.ex[b]++;
    }
    if (h.tau < N) { const ex = (S.hex[settle] ||= []); ex.push([h.tau, p.S[h.tau], p.ell[h.tau]]); if (ex.length > 4000) ex.shift(); }
  }
  function record(g, h) { recordGame(g, S.shadow); recordHolder(h, S.path); }
  function advance() {
    if (S.settled) return;
    S.n = Math.min(N, S.n + 1);
    if (S.n >= tau()) { S.settled = true; S.settleT = performance.now(); S.rounds++; record(S.game, S.hold); }
    S.dirty = true;
  }
  function replay() { compute(); S.n = 0; S.settled = false; S.dirty = true; }

  function batch(rounds) {
    if (S.batching) return;
    S.batching = true;
    let done = 0;
    const chunk = () => {
      const t0 = performance.now();
      while (done < rounds && performance.now() - t0 < 30) {
        const p = E.simulate(S.rng);
        if (S.mode !== "holder") recordGame(play(S.hedge, S.stop, p), S.mode === "game" ? play(shadowKind(), S.stop, p) : null);
        recordHolder(E.holderRound(S.settle, p), p);
        done++; S.rounds++;
      }
      $("status").textContent = `Playing ${rounds} rounds… ${done}`;
      S.dirty = true;
      if (done < rounds) requestAnimationFrame(chunk); else { S.batching = false; S.dirty = true; }
    };
    chunk();
  }

  // ---------------- canvas helpers ----------------
  function canvas(id) {
    const cv = $(id);
    const w = cv.clientWidth;
    if (!w) return null;
    const h = +(cv.dataset.h ||= cv.getAttribute("height"));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv._w !== w || cv._dpr !== dpr) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + "px";
      cv._w = w; cv._dpr = dpr;
    }
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    return { c, w, h };
  }
  function ticks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function frame(c, x0, x1, y0, y1, X, Y, opt = {}) {      // grid, y ticks, x ticks
    c.font = MONO(10.5); c.textBaseline = "middle";
    for (const v of opt.yt || []) {
      const y = Y(v);
      c.strokeStyle = C.grid; c.lineWidth = 1; c.beginPath(); c.moveTo(x0, Math.round(y) + 0.5); c.lineTo(x1, Math.round(y) + 0.5); c.stroke();
      c.fillStyle = C["ink-3"]; c.textAlign = "right"; c.fillText(opt.yf ? opt.yf(v) : v, x0 - 6, y);
    }
    c.textBaseline = "top"; c.textAlign = "center";
    for (const v of opt.xt || []) { c.fillStyle = C["ink-3"]; c.fillText(opt.xf ? opt.xf(v) : v, X(v), y1 + 6); }
    c.strokeStyle = C.line; c.beginPath(); c.moveTo(x0, Math.round(y1) + 0.5); c.lineTo(x1, Math.round(y1) + 0.5); c.stroke();
  }
  function line(c, pts, color, width = 2, dash = null) {
    if (pts.length < 2) return;
    c.save(); c.strokeStyle = color; c.lineWidth = width; c.lineJoin = c.lineCap = "round";
    if (dash) c.setLineDash(dash);
    c.beginPath(); pts.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke(); c.restore();
  }
  function dot(c, x, y, r, fill, ring = C.panel) {
    c.beginPath(); c.arc(x, y, r + 2, 0, 7); c.fillStyle = ring; c.fill();
    c.beginPath(); c.arc(x, y, r, 0, 7); c.fillStyle = fill; c.fill();
  }
  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16)), pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
    return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
  }
  const liqColor = (ell) => mix(C["liq-lo"], C["liq-hi"], Math.min(1, Math.log(ell) / Math.log(10)));
  function tip(chartId, html, x, y) {
    const host = $(chartId);
    let el = host.querySelector(".tip");
    if (!html) { if (el) el.hidden = true; return; }
    if (!el) { el = document.createElement("div"); el.className = "tip"; host.appendChild(el); }
    el.hidden = false; el.innerHTML = html;
    const w = host.clientWidth, tw = el.offsetWidth;
    el.style.left = Math.max(0, Math.min(w - tw, x + 12 - (x > w / 2 ? tw + 24 : 0))) + "px";
    el.style.top = Math.max(0, y - 10) + "px";
  }
  function hoverable(chartId) {
    const host = $(chartId);
    const set = (e) => { const r = host.getBoundingClientRect(); S.hover[chartId] = { x: e.clientX - r.left, y: e.clientY - r.top }; S.dirty = true; };
    host.addEventListener("pointermove", set);
    host.addEventListener("pointerdown", set);
    host.addEventListener("pointerleave", () => { delete S.hover[chartId]; tip(chartId, null); S.dirty = true; });
  }

  // ---------------- market ----------------
  function renderMarket() {
    const p = S.path, n = S.n, g = S.game, h = S.hold;
    $("m-S").textContent = money(p.S[n]);
    $("m-days").textContent = DAYS - dayOf(n);
    $("m-ell").innerHTML = `${p.ell[n].toFixed(1)}<small>× normal</small>`;
    $("m-kappa").textContent = money(E.liqCost(1, p.S[n], p.ell[n], p.vol[n], p.sig[n]), 2);
    const k = canvas("cv-market"); if (!k) return;
    const { c, w, h: H } = k;
    const stripH = 12;
    const x0 = 46, x1 = w - 84, y0 = 14, y1 = H - 46, sy = y1 + 26;
    let lo = Math.min(...p.S, MK.K), hi = Math.max(...p.S, MK.K);
    const pad = (hi - lo) * 0.08 + 1; lo -= pad; hi += pad;
    const X = (i) => x0 + (x1 - x0) * i / N, Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(lo, hi, 5), yf: (v) => "$" + v, xt: w < 520 ? [0, 25, 50] : [0, 10, 20, 30, 40, 50], xf: (i) => "day " + dayOf(i) });
    // strike
    line(c, [[x0, Y(MK.K)], [x1, Y(MK.K)]], C["ink-3"], 1, [5, 4]);
    c.fillStyle = C["ink-2"]; c.font = MONO(10.5); c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText("strike $" + MK.K, x1 + 6, Y(MK.K));
    // path
    const shown = S.settled ? tau() : n;
    if (S.settled && shown < N) line(c, p.S.slice(shown).map((v, i) => [X(shown + i), Y(v)]), C["ink-3"], 1.2, [2, 4]);
    line(c, p.S.slice(0, n + 1).map((v, i) => [X(i), Y(v)]), C.ink, 2);
    // jumps
    c.fillStyle = C["ink-2"];
    for (let i = 1; i <= n; i++) if (p.Nj[i - 1] > 0) {
      const x = X(i), y = Math.min(Y(p.S[i - 1]), Y(p.S[i])) - 9;
      c.beginPath(); c.moveTo(x - 4, y - 4); c.lineTo(x + 4, y - 4); c.lineTo(x, y + 2); c.closePath(); c.fill();
    }
    // exercise
    if (S.settled) {
      const t = tau(), x = X(t), y = Y(p.S[t]);
      const paid = S.mode === "holder" ? h.payoff > 0 : p.S[t] < MK.K;
      line(c, [[x, y0], [x, y1]], paid ? C.holder : C["ink-3"], 1.5);
      dot(c, x, y, 5, paid ? C.holder : C["ink-3"]);
      c.fillStyle = C.ink; c.font = FONT(11.5, 500); c.textBaseline = "top";
      const lab = paid ? `exercised · day ${dayOf(t)}` : "expired worthless";
      c.textAlign = x > (x0 + x1) / 2 ? "right" : "left";
      c.fillText(lab, x + (c.textAlign === "right" ? -8 : 8), y0 + 2);
    } else dot(c, X(n), Y(p.S[n]), 4, C.ink);
    if (S.mode === "game" && S.shadow) {           // where the same holder exercises against the comparison hedge
      const ts = S.shadow.tau;
      if ((S.settled || n >= ts) && ts !== tau() && p.S[ts] < MK.K) {
        const x = X(ts), y = Y(p.S[ts]);
        c.save(); c.translate(x, y); c.rotate(Math.PI / 4); c.fillStyle = C.panel; c.fillRect(-6, -6, 12, 12);
        c.fillStyle = C.third; c.fillRect(-4, -4, 8, 8); c.restore();
      }
    }
    // liquidity strip
    const cw = (x1 - x0) / (N + 1);
    for (let i = 0; i <= N; i++) {
      c.fillStyle = i <= n ? liqColor(p.ell[i]) : C.grid;
      c.fillRect(x0 + (x1 - x0) * i / (N + 1) + 1, sy, Math.max(1, cw - 2), stripH);
    }
    c.fillStyle = C["ink-3"]; c.font = MONO(10.5); c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText("illiquidity", x1 + 6, sy + stripH / 2);
    // hover
    const hv = S.hover["m-chart"];
    if (hv && hv.x >= x0 - 6 && hv.x <= x1 + 6) {
      const i = Math.max(0, Math.min(S.settled ? N : n, Math.round((hv.x - x0) / (x1 - x0) * N)));
      line(c, [[X(i), y0], [X(i), y1]], C["ink-3"], 1);
      dot(c, X(i), Y(p.S[i]), 4, C.ink);
      let html = `<b>Day ${dayOf(i)}</b><br>Stock ${money(p.S[i])}<br>Illiquidity ${p.ell[i].toFixed(2)}×`;
      if (p.Nj[i - 1] > 0) html += `<br>Jump: ${p.Nj[i - 1]} that step`;
      if (S.mode !== "holder" && i < N) html += `<br>Seller holds ${num(g.ro.delta[i])} sh`;
      if (S.mode !== "holder" && i >= 1) html += `<br>Seller owes if exercised ${money(g.ro.L[i])}`;
      if (S.mode === "holder" && i >= 1) html += `<br>Exercise value ${money(h.Y[i])}` + (i < N ? `<br>Value of waiting ${money(h.C[i])}` : "");
      tip("m-chart", html, X(i), hv.y);
    } else tip("m-chart", null);
    // banner
    let ban = $("m-banner");
    if (S.settled && performance.now() - S.settleT < 2600 / S.speed + 400) {
      if (!ban) { ban = document.createElement("div"); ban.id = "m-banner"; ban.className = "banner"; $("c-market").appendChild(ban); }
      const t = tau();
      if (S.mode === "holder") {
        ban.className = "banner";
        ban.textContent = h.payoff > 0 ? `Exercised on day ${dayOf(t)} · payoff ${money(h.payoff)}` + (h.kappa[t] > 0 ? ` after ${money(h.kappa[t])} liquidity cost` : "") : "Expired worthless · payoff $0";
      } else {
        const loss = g.loss;
        ban.className = loss > 0 ? "banner" : "banner seller";
        ban.textContent = (p.S[t] < MK.K ? `Holder exercises on day ${dayOf(t)}` : "Expires out of the money") + ` · seller ${loss > 0 ? "loses" : "gains"} ${money(Math.abs(loss))} before the premium`;
        if (S.mode === "game" && S.shadow) {
          const sh = S.shadow, d = sh.loss - loss;
          ban.textContent = (p.S[t] < MK.K ? `Exercised on day ${dayOf(t)}` : "Expired out of the money") +
            ` · loss before premium: ${nameOf(HEDGES, S.hedge).name} ${money(loss)}, ${nameOf(HEDGES, shadowKind()).name} ${money(sh.loss)}` +
            (sh.tau !== t ? ` (exercised day ${dayOf(sh.tau)})` : "") + ` · ${Math.abs(d) < 0.005 ? "a tie" : `${money(Math.abs(d))} ${d > 0 ? "in " + nameOf(HEDGES, S.hedge).name + "'s favour" : "in " + nameOf(HEDGES, shadowKind()).name + "'s favour"}`}`;
        }
      }
      ban.style.top = ($("m-chart").offsetTop + 10) + "px";
      ban.hidden = false;
    } else if (ban) ban.hidden = true;
  }

  // ---------------- game: seller and holder cards ----------------
  function gauge(id, v, ref) {
    const el = $(id), pos = (x) => `${Math.max(0, Math.min(1, x + 1)) * 100}%`;
    const main = el.querySelector(".mark.main"), r = el.querySelector(".mark.ref"), fill = el.querySelector(".fill"), zero = el.querySelector(".zero");
    main.style.left = pos(v); main.style.background = C.seller; r.style.left = pos(ref); zero.style.left = "100%";
    const a = Math.max(0, Math.min(1, v + 1));
    fill.style.left = a * 100 + "%"; fill.style.right = "0"; fill.style.background = C.seller; fill.style.opacity = "0.35";
  }
  function renderSeller() {
    const ro = S.game.ro, n = S.n, i = Math.min(n, N - 1);
    const d = S.hedge === "none" ? 0 : ro.delta[i];
    $("s-delta").textContent = num(d, 2);
    gauge("s-gauge", d, S.ref.delta[i]);
    const sk = shadowKind(), sd = S.shadow.ro.delta[i];
    const sm = $("s-gauge").querySelector(".mark.shadow");
    sm.style.left = `${Math.max(0, Math.min(1, sd + 1)) * 100}%`; sm.style.background = C.third;
    $("s-shadow-key").textContent = nameOf(HEDGES, sk).name;
    $("s-shadow-label").textContent = `${nameOf(HEDGES, sk).name} would owe`;
    $("s-shadow-owes").textContent = n >= 1 ? money(S.shadow.ro.L[n]) : "—";
    $("s-trade").textContent = n < N ? `${ro.trade[n] >= 0 ? "+" : "−"}${Math.abs(ro.trade[n]).toFixed(3)} sh · ${money(ro.cost[n], 3)}` : "—";
    $("s-cost").textContent = money(n < N ? ro.C[n + 1] : ro.C[N], 2);
    $("s-owes").textContent = n >= 1 ? money(ro.L[n]) : "—";
    $("s-desc").textContent = nameOf(HEDGES, S.hedge).desc;
  }
  function renderHolder() {
    const g = S.game, n = S.n, st = nameOf(STOPS, S.stop);
    $("h-role").textContent = st.role;
    $("h-desc").textContent = st.desc;
    const ex = S.settled && n === g.tau && S.path.S[n] < MK.K;
    const v = $("h-verdict");
    v.textContent = ex ? "Exercise" : S.settled && n === N ? "Expires" : n === 0 ? "Wait" : "Wait";
    v.className = "verdict " + (ex ? "go" : "wait");
    $("h-owes").textContent = n >= 1 ? money(g.ro.L[n]) : "—";
    const itm = MK.K - S.path.S[n];
    $("h-itm").textContent = itm > 0 ? `${money(itm)} in the money` : `${money(-itm)} out of the money`;
    $("h-days").textContent = dayOf(n) + " of " + DAYS;
    const rule = S.stop === "lsm_rule" || S.stop === "never";
    $("h-signal-label").textContent = rule ? "Exercise rule · 1 = exercise, −1 = wait" : "Exercise signal · exercises when it crosses 0";
    const k = canvas("cv-signal"); if (!k) return;
    const { c, w, h } = k, x0 = 30, x1 = w - 6, y0 = 6, y1 = h - 18;
    const sc = (v) => Math.asinh(Math.max(-40, Math.min(40, v)) / 2), lo = sc(-40), hi = sc(40);
    const X = (i) => x0 + (x1 - x0) * (i - 1) / (N - 1), Y = (v) => y1 - (y1 - y0) * (sc(v) - lo) / (hi - lo);
    frame(c, x0, x1, y0, y1, X, Y, { yt: rule ? [-1, 0, 1] : [-30, -5, 0, 5, 30], xt: [] });
    c.fillStyle = "rgba(232,104,63,0.10)"; c.fillRect(x0, y0, x1 - x0, Y(0) - y0);
    line(c, [[x0, Y(0)], [x1, Y(0)]], C["ink-3"], 1, [4, 3]);
    const pts = [];
    for (let i = 1; i <= Math.max(1, n); i++) if (i <= n) pts.push([X(i), Y(g.logits[i])]);
    line(c, pts, C.holder, 2);
    if (n >= 1) dot(c, X(n), Y(g.logits[n]), 3.5, C.holder);
    c.fillStyle = C["ink-3"]; c.font = MONO(10); c.textAlign = "right"; c.textBaseline = "top";
    c.fillText("exercise zone", x1 - 2, y0 + 2);
  }

  // ---------------- payoff matrix ----------------
  function buildMatrix() {
    const M = SET().matrix, H = SET().h_keys, Sk = SET().s_keys;
    const vals = H.flatMap((h) => Sk.map((s) => M[`${h}|${s}`].cvar));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const rowMax = Object.fromEntries(H.map((h) => [h, Math.max(...Sk.map((s) => M[`${h}|${s}`].cvar))]));
    const colMin = Object.fromEntries(Sk.map((s) => [s, Math.min(...H.map((h) => M[`${h}|${s}`].cvar))]));
    const minimax = H.reduce((a, b) => (rowMax[a] <= rowMax[b] ? a : b));
    let html = `<thead><tr><th class="rowh"><span class="label">Seller ↓ · Holder →</span></th>` +
      Sk.map((s) => `<th scope="col">${nameOf(STOPS, s).name}</th>`).join("") + `<th scope="col">Worst case</th></tr></thead><tbody>`;
    for (const h of H) {
      html += `<tr class="${h === minimax ? "minimax" : ""}"><th class="rowh" scope="row"><i class="dot"></i>${nameOf(HEDGES, h).name}</th>`;
      for (const s of Sk) {
        const v = M[`${h}|${s}`].cvar, t = Math.pow((v - lo) / (hi - lo), 0.6);
        const best = v === rowMax[h], saddle = best && v === colMin[s];
        html += `<td><button data-h="${h}" data-s="${s}" class="${best ? "best" : ""}" style="background:${mix(C["panel-2"], C.holder, 0.08 + 0.5 * t)}" aria-label="${nameOf(HEDGES, h).name} against ${nameOf(STOPS, s).name}: ${v.toFixed(2)}${best ? ", holder's best reply" : ""}${saddle ? ", saddle point" : ""}">${v.toFixed(2)}${saddle ? '<span class="star">★</span>' : ""}</button></td>`;
      }
      const d = SET().duals[h];
      html += `<td class="worst">${rowMax[h].toFixed(2)}${d ? `<div class="label" style="font-size:9.5px">≤ ${d.D.toFixed(2)} certified</div>` : ""}</td></tr>`;
    }
    $("mx").innerHTML = html + "</tbody>";
    $("mx-sub").textContent = `Seller's tail loss · ${cvarName()} · $ per option · 262,144 test paths`;
    // describe the saddle points and the seller's best guarantee at this level
    const saddles = [];
    for (const h of H) for (const st of Sk) { const v = M[`${h}|${st}`].cvar; if (v === rowMax[h] && v === colMin[st]) saddles.push([h, st, v]); }
    const sd = saddles.map(([h, st, v]) => `${nameOf(HEDGES, h).name} against ${nameOf(STOPS, st).name} (${v.toFixed(2)})`).join("; ");
    const dg = SET().duals.gda;
    $("mx-note").innerHTML = `Ring: the holder's best reply to each hedge. ` +
      (saddles.length ? `<b>★ Saddle point</b>: ${sd} is the worst entry in its row and the best in its column, so neither player gains by switching. `
        : `No pure saddle point among these strategies at this level: the players would want to randomise. `) +
      (minimax === "none" ? `Holding no shares at all gives the lowest worst case (${rowMax[minimax].toFixed(2)}): at this level hedging only adds trading costs` :
        `The ${nameOf(HEDGES, minimax).name}${/hedge$/.test(nameOf(HEDGES, minimax).name) ? "" : " hedge"} has the lowest worst case (${rowMax[minimax].toFixed(2)})`) +
      (dg ? `, and the dual bound certifies that no exercise rule can push the GDA hedge above ${dg.D.toFixed(2)}.` : ". No dual certificate was run at this level.");
  }
  function markMatrix() {
    for (const b of $("mx").querySelectorAll("button")) b.classList.toggle("cur", b.dataset.h === S.hedge && b.dataset.s === S.stop);
  }

  // ---------------- score (histogram) ----------------
  function renderScore() {
    const kk = key(), ref = SET().hists[mkey()], m = SET().matrix[mkey()], edges = DATA.hist_edges;
    const xs = S.live[kk] || [];
    $("sc-ref").textContent = money(m.cvar);
    $("sc-live").textContent = xs.length >= 20 ? money(Engine.cvar(xs, aNum())) : "—";
    $("sc-ref-l").textContent = `${cvarName()} · test set`; $("sc-live-l").textContent = `${cvarName()} · rounds here`;
    $("sc-tail-l").textContent = aNum() === 0 ? "all rounds count equally in the mean" : `worst ${pctTail()}% (averaged by CVaR)`;
    $("sc-n").textContent = xs.length.toLocaleString();
    const seller = S.mode === "seller";
    $("score-title").textContent = seller ? "Risk" : "Score";
    $("score-sub").textContent = seller ? `Seller's loss at exercise · against ${nameOf(STOPS, S.stop).name}` : `Seller's loss at exercise · ${nameOf(HEDGES, S.hedge).name} vs ${nameOf(STOPS, S.stop).name}`;
    const k = canvas("cv-hist"); if (!k) return;
    const { c, w, h } = k;
    let i0 = ref.findIndex((v) => v > 2e-4), i1 = ref.length - 1 - [...ref].reverse().findIndex((v) => v > 2e-4);
    i0 = Math.max(0, i0 - 1); i1 = Math.min(ref.length - 1, i1 + 1);
    const live = new Array(ref.length).fill(0);
    for (const v of xs) { const b = Math.floor((Math.min(Math.max(v, edges[0]), edges[edges.length - 1] - 1e-9) - edges[0]) / (edges[1] - edges[0])); live[b]++; }
    const liveF = live.map((v) => (xs.length ? v / xs.length : 0));
    const top = Math.max(...ref.slice(i0, i1 + 1), ...liveF.slice(i0, i1 + 1)) * 1.12;
    const x0 = 40, x1 = w - 10, y0 = 18, y1 = h - 24;
    const X = (e) => x0 + (x1 - x0) * (e - edges[i0]) / (edges[i1 + 1] - edges[i0]);
    const Y = (f) => y1 - (y1 - y0) * f / top;
    frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(0, top, 3), yf: (v) => Math.round(v * 100) + "%", xt: ticks(edges[i0], edges[i1 + 1], 6), xf: (v) => "$" + v });
    const varX = m.var;
    for (let b = i0; b <= i1; b++) {
      const xa = X(edges[b]) + 1, xb = X(edges[b + 1]) - 1;
      if (liveF[b] > 0) { c.fillStyle = edges[b] >= varX ? C.holder : C.seller; c.globalAlpha = 0.85; c.fillRect(xa, Y(liveF[b]), Math.max(1, xb - xa), y1 - Y(liveF[b])); c.globalAlpha = 1; }
    }
    // reference outline (step)
    const pts = [];
    for (let b = i0; b <= i1; b++) { pts.push([X(edges[b]), Y(ref[b])], [X(edges[b + 1]), Y(ref[b])]); }
    line(c, pts, C["ink-2"], 1.2);
    line(c, [[X(varX), y0], [X(varX), y1]], C.holder, 1, [3, 3]);
    line(c, [[X(m.cvar), y0], [X(m.cvar), y1]], C.holder, 2);
    c.font = MONO(10.5); c.fillStyle = C.ink; c.textBaseline = "bottom"; c.textAlign = "left";
    c.fillText(`${aNum() === 0 ? "mean" : "CVaR"} ${m.cvar.toFixed(2)}`, Math.min(X(m.cvar) + 4, x1 - 80), y0 + 10);
    c.fillStyle = C["ink-2"]; c.textAlign = "right";
    if (aNum() > 0) c.fillText(`worst ${pctTail()}% →`, X(varX) - 4, y0 + 10);
    const hv = S.hover["sc-chart"];
    if (hv && hv.x >= x0 && hv.x <= x1) {
      const b = Math.max(i0, Math.min(i1, i0 + Math.floor((hv.x - x0) / (x1 - x0) * (i1 - i0 + 1))));
      tip("sc-chart", `Loss ${money(edges[b], 1)} to ${money(edges[b + 1], 1)}<br>Rounds here ${(liveF[b] * 100).toFixed(1)}%<br>Test paths ${(ref[b] * 100).toFixed(1)}%`, hv.x, hv.y);
    } else tip("sc-chart", null);
    if (seller) renderSpark();
  }
  function renderSpark() {
    const log = S.hedge === "gda" ? SET().logs.gda.map((r) => [r.it, r.hard_cvar]) : S.hedge === "lsm" ? SET().logs.lsm.map((r) => [r.it, r.loss]) : null;
    $("sc-train").hidden = !log;
    if (!log) return;
    const k = canvas("cv-spark"); if (!k) return;
    const { c, w, h } = k, x0 = 34, x1 = w - 22, y0 = 6, y1 = h - 16;
    const lo = Math.min(...log.map((r) => r[1])) - 0.3, hi = Math.max(...log.map((r) => r[1])) + 0.3;
    const itMax = log[log.length - 1][0] + 1;
    const X = (i) => x0 + (x1 - x0) * i / itMax, Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    frame(c, x0, x1, y0, y1, X, Y, { yt: [Math.ceil(lo), Math.floor(hi)], xt: [0, Math.round(itMax / 2), itMax], xf: (v) => v ? `it ${v.toLocaleString()}` : "0" });
    line(c, log.map(([i, v]) => [X(i), Y(v)]), C.seller, 2);
    const [li, lv] = log[log.length - 1]; dot(c, X(li), Y(lv), 3.5, C.seller);
  }

  // ---------------- training replay ----------------
  function renderTrain(now) {
    const st = SET(), log = st.logs.gda, d = st.duals.gda, eq = st.matrix["gda|advA"].cvar;
    const k = canvas("cv-train"); if (!k) return;
    const { c, w, h } = k, x0 = 36, x1 = w - 20, y0 = 10, y1 = h - 22;
    const itMax = log[log.length - 1].it + 1;
    const vals = log.flatMap((r) => [r.hedger_loss, r.stopper_obj]).filter((v) => v != null);
    const lo = Math.floor(Math.min(...vals, eq) - 0.5), hi = Math.ceil(Math.max(...vals, d ? d.D : eq) + 0.5);
    const X = (i) => x0 + (x1 - x0) * Math.sqrt(Math.max(0, i) / itMax), Y = (v) => y1 - (y1 - y0) * (Math.min(hi, v) - lo) / (hi - lo);
    const xt = itMax > 1000 ? [0, 100, 500, 1000, 1500] : [0, 50, 200, 400, 600];
    frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(lo, hi, 5), yf: (v) => "$" + v, xt, xf: (v) => (v ? v.toLocaleString() : "it 0") });
    if (d) { c.fillStyle = C.third; c.globalAlpha = 0.22; c.fillRect(x0, Y(d.D), x1 - x0, Y(eq) - Y(d.D)); c.globalAlpha = 1; }
    const cyc = 26000, it = reduced ? itMax - 1 : Math.min(itMax - 1, ((now % cyc) / (cyc - 3000)) * itMax);
    const upTo = (f) => { const pts = []; for (const r of log) { if (r.it <= it && r[f] != null) pts.push([X(r.it), Y(r[f])]); } return pts; };
    const lerp = (f) => { let a = log[0], b = log[log.length - 1]; for (let j = 1; j < log.length; j++) if (log[j].it >= it) { a = log[j - 1]; b = log[j]; break; } const t = b.it > a.it ? (it - a.it) / (b.it - a.it) : 0; return a[f] + (b[f] - a[f]) * Math.max(0, Math.min(1, t)); };
    const hs = upTo("hedger_loss"), as = upTo("stopper_obj");
    hs.push([X(it), Y(lerp("hedger_loss"))]); as.push([X(it), Y(lerp("stopper_obj"))]);
    line(c, as, C.holder, 2); line(c, hs, C.seller, 2);
    line(c, [[X(it), y0], [X(it), y1]], C["ink-3"], 1);
    dot(c, X(it), Y(lerp("stopper_obj")), 3.5, C.holder); dot(c, X(it), Y(lerp("hedger_loss")), 3.5, C.seller);
    c.fillStyle = C.ink; c.font = MONO(10.5); c.textAlign = X(it) > (x0 + x1) / 2 ? "right" : "left"; c.textBaseline = "top";
    c.fillText(`iteration ${Math.round(it)}`, X(it) + (c.textAlign === "right" ? -6 : 6), y0 + 2);
    $("tr-sub").textContent = `Gradient descent–ascent · ${itMax.toLocaleString()} iterations${st.restarts ? " · with attacker restarts" : ""} · early ones stretched`;
    $("tr-band").hidden = !d;
    $("tr-note").innerHTML = `Each iteration the attacker takes five steps to raise the seller's ${aNum() === 0 ? "expected loss" : "tail loss"}, then the seller takes one step to cut it. ` +
      `Starting near $${Math.round(log[0].hedger_loss)}, they meet at about <b>$${eq.toFixed(2)}</b> (the GDA hedge against its own attacker on the test paths).` +
      (d ? ` The blue band runs from there to the certified ceiling of $${d.D.toFixed(2)}.` : "") +
      (S.alpha !== "0.9" ? ` This level comes from the shorter grid run.` : "");
  }

  // ---------------- head to head: the chosen hedge against the comparison hedge, same paths ----------------
  function pairedCvar(a, b) {          // CVaR(a) - CVaR(b) on paired samples, influence-function SE (losses.paired_cvar_diff)
    const AL = aNum(), n = a.length, k = Math.max(1, Math.min(n, Math.ceil((1 - AL) * n - 1e-9)));
    const va = Float64Array.from(a).sort()[n - k], vb = Float64Array.from(b).sort()[n - k];
    const d = AL === 0 ? a.map((x, i) => x - b[i]) : a.map((x, i) => (va + Math.max(x - va, 0) / (1 - AL)) - (vb + Math.max(b[i] - vb, 0) / (1 - AL)));
    const m = d.reduce((s, v) => s + v, 0) / n, sd = Math.sqrt(d.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, n - 1));
    return [Engine.cvar(a, AL) - Engine.cvar(b, AL), sd / Math.sqrt(n)];
  }
  function renderDuel() {
    const sk = shadowKind(), mainName = nameOf(HEDGES, S.hedge).name, shName = nameOf(HEDGES, sk).name;
    const rows = S.duel[duelKey()] || [], n = rows.length;
    const a = rows.map((r) => r[0]), b = rows.map((r) => r[1]);
    const refA = SET().matrix[`${S.hedge}|${S.stop}`].cvar, refB = SET().matrix[`${sk}|${S.stop}`].cvar;
    for (const id of ["du-a-cvar-l", "du-b-cvar-l"]) { $(id).textContent = `${cvarName()} · rounds here`; $(id).parentElement.hidden = aNum() === 0; }
    for (const id of ["du-a-ref-l", "du-b-ref-l"]) $(id).textContent = `${cvarName()} · test paths`;
    $("du-sub").textContent = `Same paths, same holder (${nameOf(STOPS, S.stop).name}) · seller's loss before the premium`;
    $("du-a-name").textContent = mainName; $("du-b-name").textContent = shName;
    const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
    $("du-a-mean").textContent = n ? money(mean(a)) : "—"; $("du-b-mean").textContent = n ? money(mean(b)) : "—";
    $("du-a-cvar").textContent = n >= 20 ? money(Engine.cvar(a, aNum())) : "—"; $("du-b-cvar").textContent = n >= 20 ? money(Engine.cvar(b, aNum())) : "—";
    $("du-a-ref").textContent = money(refA); $("du-b-ref").textContent = money(refB);
    const refD = refB - refA;
    let live = "—";
    if (n >= 20) { const [d, se] = pairedCvar(b, a); live = `${d >= 0 ? "+" : "−"}${money(Math.abs(d))} ± ${(2 * se).toFixed(2)}`; }
    $("du-diff-label").textContent = `${shName} minus ${mainName}, ${cvarName()}`;
    $("du-tail-l").textContent = aNum() === 0 ? "" : `each hedge's worst ${pctTail()}%`;
    $("du-tail-l").parentElement.hidden = aNum() === 0;
    $("du-diff-live").textContent = `${live}${n ? ` · ${n.toLocaleString()} rounds` : ""}`;
    $("du-diff-ref").textContent = `${refD >= 0 ? "+" : "−"}${money(Math.abs(refD))} on 262,144 test paths`;
    const worse = n ? rows.filter((r) => r[1] - r[0] > 0.005).length / n : 0, better = n ? rows.filter((r) => r[0] - r[1] > 0.005).length / n : 0;
    const freq = n ? (better > worse
      ? ` In the rounds played here the ${shName} hedge actually lost less ${Math.round(better * 100)}% of the time, and more only ${Math.round(worse * 100)}%.`
      : ` In the rounds played here the ${shName} hedge lost more ${Math.round(worse * 100)}% of the time and less ${Math.round(better * 100)}%.`) : "";
    $("du-note").innerHTML = (refD > 0
      ? (aNum() === 0 ? `With α = 0 the score is the plain average loss.${freq} On the test paths the ${shName} hedge's average loss is <b>${money(refD)}</b> higher.`
        : `On most paths the two hedges lose about the same, so the dots hug the line.${freq} What matters is the tail, the worst ${pctTail()}% of rounds that CVaR averages: there the ${shName} hedge loses about <b>${money(refD)}</b> more per option on the test paths.`)
      : `Against this holder the ${shName} hedge does slightly better, by ${money(-refD)}: ${sk === "lsm" ? "it was trained for exactly this kind of holder. Switch the holder to an attacker to see the gap open." : "this is the holder the LSM-trained hedge was built for."}${freq}`) +
      ` The live estimate is noisy below a few thousand rounds; press B to play 1,000 more.`;
    // recent rounds
    $("du-recent").innerHTML = rows.slice(-5).reverse().map((r) => {
      const d = r[1] - r[0];
      return `<tr><td>day ${dayOf(r[2])}${r[3] !== r[2] ? ` / ${dayOf(r[3])}` : ""}</td><td>${money(r[0])}</td><td>${money(r[1])}</td><td class="${d > 0.005 ? "pos" : d < -0.005 ? "neg" : ""}">${Math.abs(d) < 0.005 ? "tie" : (d > 0 ? "+" : "−") + money(Math.abs(d))}</td></tr>`;
    }).join("") || `<tr><td colspan="4">No rounds yet</td></tr>`;
    $("du-th-a").textContent = mainName; $("du-th-b").textContent = shName;
    // scatter: each dot is one round
    const k = canvas("cv-duel"); if (!k) return;
    const { c, w, h } = k, x0 = 44, x1 = w - 12, y0 = 10, y1 = h - 26;
    const pts = rows.slice(-3000);
    let lo = -2, hi = 20;
    if (pts.length) {
      const all = pts.flatMap((r) => [r[0], r[1]]).sort((p, q) => p - q);
      lo = Math.min(-1, all[Math.floor(all.length * 0.002)]); hi = Math.max(10, all[Math.floor(all.length * 0.998)]);
    }
    const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    const X = (v) => x0 + (x1 - x0) * (v - lo) / (hi - lo), Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(lo, hi, 4), yf: (v) => "$" + v, xt: ticks(lo, hi, w < 420 ? 4 : 6), xf: (v) => "$" + v });
    line(c, [[X(lo), Y(lo)], [X(hi), Y(hi)]], C["ink-3"], 1, [4, 4]);
    // the 90% quantile of each hedge's loss: beyond these lines is the tail CVaR averages
    if (n >= 20 && aNum() > 0) {
      const qa = Float64Array.from(a).sort()[Math.floor(aNum() * n)], qb = Float64Array.from(b).sort()[Math.floor(aNum() * n)];
      c.fillStyle = C.holder; c.globalAlpha = 0.06;
      c.fillRect(X(qa), y0, x1 - X(qa), y1 - y0); c.fillRect(x0, y0, x1 - x0, Y(qb) - y0); c.globalAlpha = 1;
    }
    for (const r of pts) {
      const d = r[1] - r[0];
      c.beginPath(); c.arc(X(Math.min(hi, Math.max(lo, r[0]))), Y(Math.min(hi, Math.max(lo, r[1]))), 2.4, 0, 7);
      c.fillStyle = d > 0.005 ? C.third : d < -0.005 ? C.seller : C["ink-3"]; c.globalAlpha = 0.75; c.fill(); c.globalAlpha = 1;
    }
    if (pts.length) { const r = pts[pts.length - 1]; dot(c, X(Math.min(hi, Math.max(lo, r[0]))), Y(Math.min(hi, Math.max(lo, r[1]))), 4, C.ink); }
    c.fillStyle = C["ink-3"]; c.font = MONO(10); c.textAlign = "left"; c.textBaseline = "top";
    c.fillText(`${shName} lost more ↑`, x0 + 6, y0 + 2);
    c.textAlign = "right"; c.textBaseline = "bottom"; c.fillText(`${mainName} lost more →`, x1 - 4, y1 - 4);
    const hv = S.hover["du-chart"];
    if (hv && pts.length && hv.x >= x0 && hv.x <= x1 && hv.y >= y0 && hv.y <= y1) {
      let best = null, bd = 144;
      for (const r of pts) { const dx = X(r[0]) - hv.x, dy = Y(r[1]) - hv.y, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = r; } }
      if (best) tip("du-chart", `${mainName} ${money(best[0])}<br>${shName} ${money(best[1])}<br>Exercised day ${dayOf(best[2])}${best[3] !== best[2] ? ` / ${dayOf(best[3])}` : ""}`, X(best[0]), Y(best[1]));
      else tip("du-chart", null);
    } else tip("du-chart", null);
  }

  // ---------------- seller mode ----------------
  function renderHedge() {
    const ro = S.game.ro, ref = S.ref, n = S.n, i = Math.min(n, N - 1);
    $("hd-delta").textContent = num(S.hedge === "none" ? 0 : ro.delta[i]);
    $("hd-ref").textContent = num(ref.delta[i]);
    $("hd-trade").textContent = n < N ? `Trade ${ro.trade[n] >= 0 ? "+" : "−"}${Math.abs(ro.trade[n]).toFixed(3)} sh · cost ${money(ro.cost[n], 3)}` : "No trade at expiry";
    let k = canvas("cv-shares");
    if (k) {
      const { c, w, h } = k, x0 = 40, x1 = w - 26, y0 = 10, y1 = h - 22, lo = -1.1, hi = 0.1;
      const X = (j) => x0 + (x1 - x0) * j / N, Y = (v) => y1 - (y1 - y0) * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo);
      frame(c, x0, x1, y0, y1, X, Y, { yt: [-1, -0.5, 0], yf: (v) => num(v, 1), xt: [0, 25, 50], xf: (j) => "day " + dayOf(j) });
      const upto = Math.min(n, N - 1);
      line(c, Array.from({ length: upto + 1 }, (_, j) => [X(j), Y(ref.delta[j])]), C["ink-2"], 1.5, [5, 4]);
      line(c, Array.from({ length: upto + 1 }, (_, j) => [X(j), Y(S.hedge === "none" ? 0 : ro.delta[j])]), C.seller, 2);
      dot(c, X(upto), Y(ref.delta[upto]), 3, C["ink-2"]);
      dot(c, X(upto), Y(S.hedge === "none" ? 0 : ro.delta[upto]), 4, C.seller);
      c.fillStyle = C["ink-3"]; c.font = MONO(10); c.textAlign = "left"; c.textBaseline = "top"; c.fillText("shares held", x0 + 4, y0);
    }
    k = canvas("cv-costs");
    if (k) {
      const { c, w, h } = k, a = ro.C[Math.min(n + 1, N)], b = ref.C[Math.min(n + 1, N)];
      const top = Math.max(ro.C[N], ref.C[N], 0.01), x0 = 96, x1 = w - 60;
      const rows = [["this hedge", a, C.seller], ["Merton delta", b, C["ink-2"]]];
      rows.forEach(([lab, v, col], r) => {
        const y = 8 + r * 28;
        c.fillStyle = C.grid; c.fillRect(x0, y, x1 - x0, 14);
        c.fillStyle = col; c.fillRect(x0, y, (x1 - x0) * v / top, 14);
        c.fillStyle = C["ink-2"]; c.font = FONT(12); c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(lab, x0 - 8, y + 7);
        c.textAlign = "left"; c.fillStyle = C.ink; c.fillText(money(v), x1 + 8, y + 7);
      });
    }
  }
  function renderSPolicy() {
    const k = canvas("cv-spolicy"); if (!k) return;
    const p = S.path, n = Math.min(S.n, N - 1), ro = S.game.ro, dPrev = n > 0 ? ro.delta[n - 1] : 0;
    const { c, w, h } = k, x0 = 40, x1 = w - 20, y0 = 10, y1 = h - 24, lo = -1.1, hi = 0.1, s0 = 70, s1 = 130;
    const X = (s) => x0 + (x1 - x0) * (s - s0) / (s1 - s0), Y = (v) => y1 - (y1 - y0) * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo);
    frame(c, x0, x1, y0, y1, X, Y, { yt: [-1, -0.5, 0], yf: (v) => num(v, 1), xt: w < 420 ? [70, 100, 130] : [70, 85, 100, 115, 130], xf: (s) => "$" + s });
    line(c, [[X(MK.K), y0], [X(MK.K), y1]], C.grid, 1);
    const grid = []; for (let s = s0; s <= s1 + 1e-9; s += 1) grid.push(s);
    const at = (s, liqState) => {
      const q = { S: [], ell: [], vol: [], sig: [] };
      q.S[n] = s; q.ell[n] = liqState.ell; q.vol[n] = liqState.vol; q.sig[n] = liqState.sig;
      return E.hedgeAt(kindOf(S.hedge), q, n, dPrev);
    };
    const today = { ell: p.ell[n], vol: p.vol[n], sig: p.sig[n] }, normal = { ell: 1, vol: 1, sig: MK.sigma };
    line(c, grid.map((s) => [X(s), Y(E.mertonDelta(s, MK.T - E.t[n]))]), C["ink-2"], 1.5, [5, 4]);
    if (S.hedge !== "delta") {
      line(c, grid.map((s) => [X(s), Y(at(s, normal))]), C.seller, 1.5, [5, 4]);
      line(c, grid.map((s) => [X(s), Y(at(s, today))]), C.seller, 2);
    }
    const cur = S.hedge === "none" ? 0 : ro.delta[n];
    dot(c, X(Math.max(s0, Math.min(s1, p.S[n]))), Y(cur), 4.5, C.seller);
    const diff = at(p.S[n], today) - at(p.S[n], normal);
    $("sp-note").innerHTML = S.hedge === "delta" || S.hedge === "none" ? nameOf(HEDGES, S.hedge).desc :
      Math.abs(diff) < 0.05 ? `Day ${dayOf(n)}: liquidity is close to normal (${p.ell[n].toFixed(1)}×), so the two curves nearly agree. After a sharp fall, illiquidity jumps and the solid curve pulls toward zero: the hedge holds fewer shares when every trade costs more.`
      : diff > 0 ? `Day ${dayOf(n)}, illiquidity ${p.ell[n].toFixed(1)}×. At today's price the hedge holds <b>${diff.toFixed(2)} fewer shares</b> short than it would in a normal market: it trades less when every share costs more.`
      : `Day ${dayOf(n)}, illiquidity ${p.ell[n].toFixed(1)}×. At today's price the hedge holds <b>${(-diff).toFixed(2)} more shares</b> short than it would in a normal market.`;
    const hv = S.hover["sp-chart"];
    if (hv && hv.x >= x0 && hv.x <= x1) {
      const s = Math.round(s0 + (hv.x - x0) / (x1 - x0) * (s1 - s0));
      line(c, [[X(s), y0], [X(s), y1]], C["ink-3"], 1);
      tip("sp-chart", `Stock ${money(s, 0)}<br>Today's liquidity ${S.hedge === "delta" ? "—" : num(at(s, today))} sh<br>Normal liquidity ${S.hedge === "delta" ? "—" : num(at(s, normal))} sh<br>Merton delta ${num(E.mertonDelta(s, MK.T - E.t[n]))} sh`, X(s), hv.y);
    } else tip("sp-chart", null);
  }

  // ---------------- holder mode ----------------
  function renderDecide() {
    const hr = S.hold, n = S.n, st = nameOf(SETTLES, S.settle);
    const ex = S.settled && n === hr.tau && hr.payoff > 0;
    const v = $("dc-verdict");
    v.textContent = n === 0 ? "Wait" : ex ? "Exercise" : S.settled && n === N ? "Expires" : "Wait";
    v.className = "verdict " + (ex ? "go" : "wait");
    let k = canvas("cv-decide");
    const Y = n >= 1 ? hr.Y[n] : 0, Cv = n >= 1 && n < N ? hr.C[n] : 0;
    const intrinsic = n >= 1 ? Math.max(0, E.disc[n] * (MK.K - S.path.S[n])) : 0, kap = Math.max(0, intrinsic - Y);
    if (k) {
      const { c, w } = k, x0 = 118, x1 = w - 70;
      const top = Math.max(2, Y + kap, Cv) * 1.15;
      const bars = [["Exercise now", Y, C.holder, kap], ["Wait", Cv, C["ink-2"], 0]];
      bars.forEach(([lab, val, col, extra], r) => {
        const y = 14 + r * 58, bw = (x1 - x0);
        c.fillStyle = C.grid; c.fillRect(x0, y, bw, 22);
        c.fillStyle = col; c.fillRect(x0, y, bw * val / top, 22);
        if (extra > 0) {
          c.save(); c.strokeStyle = C.holder; c.globalAlpha = 0.7; c.lineWidth = 1;
          const xa = x0 + bw * val / top, xb = x0 + bw * (val + extra) / top;
          c.beginPath(); c.rect(xa, y, xb - xa, 22); c.clip();
          for (let x = xa - 30; x < xb + 30; x += 5) { c.beginPath(); c.moveTo(x, y + 22); c.lineTo(x + 22, y); c.stroke(); }
          c.restore();
          c.fillStyle = C["ink-2"]; c.font = FONT(11); c.textAlign = "left"; c.textBaseline = "top";
          c.fillText(`liquidity cost ${money(extra)}`, x0, y + 28);
        }
        c.fillStyle = C.ink; c.font = FONT(13, 500); c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(lab, x0 - 10, y + 11);
        c.textAlign = "left"; c.fillText(n >= 1 && (r === 0 || n < N) ? money(val) : "—", x0 + bw * Math.max(val + extra, 0) / top + 8, y + 11);
      });
    }
    k = canvas("cv-decide-hist");
    if (k) {
      const { c, w, h } = k, x0 = 36, x1 = w - 8, y0 = 6, y1 = h - 16;
      let top = 1; for (let j = 1; j <= N; j++) top = Math.max(top, hr.Y[j], j < N ? hr.C[j] : 0);
      top *= 1.1;
      const X = (j) => x0 + (x1 - x0) * (j - 1) / (N - 1), Yp = (v) => y1 - (y1 - y0) * v / top;
      frame(c, x0, x1, y0, y1, X, Yp, { yt: ticks(0, top, 2), yf: (v) => "$" + v, xt: [] });
      const upto = Math.max(1, n);
      line(c, Array.from({ length: Math.min(upto, N - 1) }, (_, j) => [X(j + 1), Yp(hr.C[j + 1])]), C["ink-2"], 1.5, [5, 4]);
      line(c, Array.from({ length: upto }, (_, j) => [X(j + 1), Yp(hr.Y[j + 1])]), C.holder, 2);
    }
    $("dc-note").innerHTML = `${st.desc} The value of waiting comes from the network fitted for this settlement; the holder exercises the first day the exercise value reaches it.`;
  }
  function meanSe(arr) {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, arr.length - 1));
    return [m, sd / Math.sqrt(arr.length)];
  }
  function renderPrice() {
    const P = DATA.pricing, sel = P[S.settle];
    $("pr-br").textContent = `$${sel.lower.toFixed(2)}–${sel.upper.toFixed(2)}`;
    const xs = S.hlive[S.settle] || [];
    if (xs.length >= 20) { const [m, se] = meanSe(xs); $("pr-live").innerHTML = `$${m.toFixed(2)}<small>±${(2 * se).toFixed(2)}</small>`; }
    else $("pr-live").textContent = "—";
    const k = canvas("cv-price");
    if (k) {
      const { c, w, h } = k, x0 = 118, x1 = w - 14, y0 = 8, y1 = h - 22;
      const all = SETTLES.flatMap((s) => [P[s.k].lower, P[s.k].upper, P[s.k].european]);
      const lo = Math.floor((Math.min(...all) - 0.35) * 2) / 2, hi = Math.ceil((Math.max(...all) + 0.35) * 2) / 2;
      const X = (v) => x0 + (x1 - x0) * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo);
      frame(c, x0, x1, y0, y1, X, () => 0, { yt: [], xt: ticks(lo, hi, w < 420 ? 3 : 5), xf: (v) => "$" + v.toFixed(1) });
      SETTLES.forEach((s, r) => {
        const q = P[s.k], y = y0 + 18 + r * ((y1 - y0 - 18) / 3), on = s.k === S.settle;
        c.fillStyle = on ? C.ink : C["ink-2"]; c.font = FONT(12.5, on ? 600 : 400); c.textAlign = "right"; c.textBaseline = "middle";
        c.fillText(s.name, x0 - 10, y);
        c.fillStyle = on ? C.ink : C["ink-3"]; c.fillRect(X(q.lower), y - 6, Math.max(3, X(q.upper) - X(q.lower)), 12);
        line(c, [[X(q.european), y - 12], [X(q.european), y + 12]], C["ink-2"], 1.5);
        const arr = S.hlive[s.k] || [];
        if (arr.length >= 20) {
          const [m, se] = meanSe(arr), a = m - 2 * se, b = m + 2 * se;
          line(c, [[X(a), y + 18], [X(b), y + 18]], C.holder, 1.5);
          if (a < lo) { c.fillStyle = C.holder; c.beginPath(); c.moveTo(x0, y + 18); c.lineTo(x0 + 6, y + 14); c.lineTo(x0 + 6, y + 22); c.fill(); }
          if (b > hi) { c.fillStyle = C.holder; c.beginPath(); c.moveTo(x1, y + 18); c.lineTo(x1 - 6, y + 14); c.lineTo(x1 - 6, y + 22); c.fill(); }
          if (m >= lo && m <= hi) dot(c, X(m), y + 18, 3.5, C.holder);
        }
      });
    }
    const cash = P.cash_lam1, prem = [Math.max(0, sel.lower - sel.european), sel.upper - sel.european];
    const disc = [cash.lower - sel.upper, cash.upper - sel.lower];
    $("pr-note").innerHTML = `Early-exercise premium over the European put: <b>$${prem[0].toFixed(2)}–${prem[1].toFixed(2)}</b>.` +
      (S.settle === "cash_lam1" ? " Cash settlement gives the classical price." : ` Liquidity discount against cash settlement: <b>$${Math.max(0, disc[0]).toFixed(2)}–${disc[1].toFixed(2)}</b>, about ${Math.round(100 * (cash.lower + cash.upper - sel.lower - sel.upper) / (cash.lower + cash.upper))}%.`) +
      ` Both ends come from 262,144 independent paths: the fitted exercise rule gives the lower end, the dual the upper. A single round's payoff is noisy, so the live average needs a few thousand rounds to land inside the bracket.`;
  }
  function renderHPolicy() {
    const pts = S.hex[S.settle] || [], pr = S.hprop[S.settle] || { alive: [0, 0, 0, 0], ex: [0, 0, 0, 0] };
    let k = canvas("cv-hpolicy");
    if (k) {
      const { c, w, h } = k, x0 = 40, x1 = w - 14, y0 = 8, y1 = h - 22, lo = 60, hi = 101;
      const X = (j) => x0 + (x1 - x0) * j / N, Y = (v) => y1 - (y1 - y0) * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo);
      frame(c, x0, x1, y0, y1, X, Y, { yt: [60, 70, 80, 90, 100], yf: (v) => "$" + v, xt: [0, 25, 50], xf: (j) => "day " + dayOf(j) });
      const order = [...pts].sort((a, b) => a[2] - b[2]);            // draw the illiquid ones on top
      for (const [j, s, ell] of order) {
        c.beginPath(); c.arc(X(j) + (((j * 7919 + s * 31) % 7) - 3) * 0.6, Y(s), 2.6, 0, 7);
        c.fillStyle = liqColor(Math.max(1.6, ell)); c.globalAlpha = 0.8; c.fill(); c.globalAlpha = 1;
      }
      const p = S.path, n = S.n;
      line(c, p.S.slice(0, n + 1).map((v, j) => [X(j), Y(v)]), C.holder, 1.4);
      dot(c, X(n), Y(p.S[n]), 3.5, C.holder);
      c.fillStyle = C["ink-3"]; c.font = MONO(10); c.textAlign = "right"; c.textBaseline = "top";
      c.fillText(`${pts.length.toLocaleString()} early exercises`, x1, y0);
    }
    k = canvas("cv-hprop");
    if (k) {
      const { c, w, h } = k, x0 = 84, x1 = w - 104, labels = ["below 1.25×", "1.25–2×", "2–4×", "above 4×"];
      const rates = pr.alive.map((a, i) => (a ? pr.ex[i] / a : 0)), top = Math.max(0.05, ...rates) * 1.1;
      labels.forEach((lab, i) => {
        const y = 4 + i * 28, bw = (x1 - x0) * rates[i] / top;
        c.fillStyle = C.grid; c.fillRect(x0, y, x1 - x0, 16);
        c.fillStyle = liqColor([1.1, 1.6, 2.8, 6][i] > 1.5 ? [1.1, 1.6, 2.8, 6][i] : 1.6); c.fillRect(x0, y, bw, 16);
        c.fillStyle = C["ink-2"]; c.font = FONT(11.5); c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(lab, x0 - 8, y + 8);
        c.textAlign = "left"; c.fillStyle = C.ink;
        c.fillText(pr.alive[i] ? `${(rates[i] * 100).toFixed(1)}%` : "—", x1 + 8, y + 8);
        c.fillStyle = C["ink-3"]; c.font = MONO(9.5);
        c.fillText(`n ${pr.alive[i].toLocaleString()}`, x1 + 8 + 44, y + 8);
      });
    }
    $("hp-note").innerHTML = S.settle === "cash_lam1"
      ? "Cash settlement: exercising costs nothing, so the decision depends only on time and price. Switch to physical settlement to see illiquidity hold the holder back."
      : "Each bar is the share of deep in-the-money days on which the holder exercised. Physically settled holders mostly exercise while the market is liquid and wait out illiquid spells. On 262,144 test paths (physical, small, stock at 70–80% of strike) the rates were 26.6%, 5.2%, 0.2% and 0.9%.";
  }

  // ================= play the holder =================
  // You exercise once per round; both sellers (GDA and LSM-trained, alpha = 0.9) settle at your date. Each is
  // compared with the attacker trained against it, on the same path.
  const PL = { rng: Engine.makeRng(424242), path: null, A: null, B: null, attA: null, attB: null, n: 0, tau: null,
    settled: false, running: false, started: false, acc: 0, rounds: [], speed: 1 };
  const PL_STEP_MS = 380;
  function plNew(run) {
    const p = E.simulate(PL.rng);
    Object.assign(PL, { path: p, A: E.rollout("gda", p), B: E.rollout("lsm", p), attA: E.playRound("gda", "advA", p),
      attB: E.playRound("lsm", "advB", p), ownB: E.playRound("lsm", "lsm_rule", p), n: 0, tau: null, settled: false, running: !!run, acc: 0 });
    if (run) PL.started = true;
    S.dirty = true;
  }
  function plSettle(t) {
    PL.tau = t; PL.settled = true; PL.running = false;
    PL.rounds.push({ t, ya: PL.A.L[t], aa: PL.attA.loss, yb: PL.B.L[t], ab: PL.attB.loss, ta: PL.attA.tau, tb: PL.attB.tau, ob: PL.ownB.loss });
    S.dirty = true;
  }
  function plStep() { if (PL.settled) return; PL.n = Math.min(N, PL.n + 1); if (PL.n >= N) plSettle(N); S.dirty = true; }
  function plExercise() { if (!PL.settled && PL.n >= 1) plSettle(PL.n); }
  function plToggle() {
    if (PL.settled) { plNew(true); return; }
    PL.running = !PL.running; PL.started = true; S.dirty = true;
  }
  function plTick(dt) {
    if (S.mode !== "play" || !PL.running || PL.settled) return;
    PL.acc += dt;
    const ms = PL_STEP_MS / PL.speed;
    while (PL.acc >= ms && !PL.settled) { PL.acc -= ms; plStep(); }
  }
  const diamond = (c, x, y, col) => { c.save(); c.translate(x, y); c.rotate(Math.PI / 4); c.fillStyle = C.panel; c.fillRect(-6, -6, 12, 12); c.fillStyle = col; c.fillRect(-4, -4, 8, 8); c.restore(); };

  function renderPlay() {
    const p = PL.path, n = PL.n, A = PL.A, B = PL.B;
    // --- market ---
    $("pm-S").textContent = money(p.S[n]);
    $("pm-days").textContent = DAYS - dayOf(n);
    $("pm-ell").innerHTML = `${p.ell[n].toFixed(1)}<small>× normal</small>`;
    $("pm-pay").textContent = money(Math.max(0, MK.K - p.S[n]));
    $("pm-sub").textContent = `Round ${PL.rounds.length + (PL.settled ? 0 : 1)} · day ${dayOf(n)} of ${DAYS}` +
      (PL.settled ? " · round over" : !PL.started ? " · press Space to start" : PL.running ? "" : " · paused");
    $("pl-start-t").textContent = PL.settled ? "Next round" : PL.running ? "Pause" : n === 0 ? "Start round" : "Resume";
    $("pl-ex").disabled = PL.settled || n < 1;
    $("pl-next").hidden = !PL.settled;
    $("pl-start").hidden = PL.settled;
    let k = canvas("cv-pmarket");
    if (k) {
      const { c, w, h: H } = k, x0 = 46, x1 = w - 84, y0 = 14, y1 = H - 46, sy = y1 + 26, stripH = 12;
      let lo = Math.min(...p.S, MK.K), hi = Math.max(...p.S, MK.K);
      const pad = (hi - lo) * 0.08 + 1; lo -= pad; hi += pad;
      const X = (i) => x0 + (x1 - x0) * i / N, Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
      frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(lo, hi, 5), yf: (v) => "$" + v, xt: w < 520 ? [0, 25, 50] : [0, 10, 20, 30, 40, 50], xf: (i) => "day " + dayOf(i) });
      line(c, [[x0, Y(MK.K)], [x1, Y(MK.K)]], C["ink-3"], 1, [5, 4]);
      c.fillStyle = C["ink-2"]; c.font = MONO(10.5); c.textAlign = "left"; c.textBaseline = "middle"; c.fillText("strike $" + MK.K, x1 + 6, Y(MK.K));
      if (PL.settled && PL.tau < N) line(c, p.S.slice(PL.tau).map((v, i) => [X(PL.tau + i), Y(v)]), C["ink-3"], 1.2, [2, 4]);
      const upto = PL.settled ? PL.tau : n;
      line(c, p.S.slice(0, upto + 1).map((v, i) => [X(i), Y(v)]), C.ink, 2);
      c.fillStyle = C["ink-2"];
      for (let i = 1; i <= upto; i++) if (p.Nj[i - 1] > 0) {
        const x = X(i), y = Math.min(Y(p.S[i - 1]), Y(p.S[i])) - 9;
        c.beginPath(); c.moveTo(x - 4, y - 4); c.lineTo(x + 4, y - 4); c.lineTo(x, y + 2); c.closePath(); c.fill();
      }
      if (PL.settled) {
        diamond(c, X(PL.attA.tau), Y(p.S[PL.attA.tau]) - 12, C.seller);
        diamond(c, X(PL.attB.tau), Y(p.S[PL.attB.tau]) + 12, C.third);
        const x = X(PL.tau), y = Y(p.S[PL.tau]);
        line(c, [[x, y0], [x, y1]], C.holder, 1.5); dot(c, x, y, 5, C.holder);
        c.fillStyle = C.ink; c.font = FONT(11.5, 500); c.textBaseline = "top"; c.textAlign = x > (x0 + x1) / 2 ? "right" : "left";
        c.fillText(`you · day ${dayOf(PL.tau)}`, x + (c.textAlign === "right" ? -8 : 8), y0 + 2);
      } else dot(c, X(n), Y(p.S[n]), 4, C.ink);
      const cw = (x1 - x0) / (N + 1);
      for (let i = 0; i <= N; i++) { c.fillStyle = i <= upto || PL.settled ? liqColor(p.ell[i]) : C.grid; if (!PL.settled && i > n) c.fillStyle = C.grid; c.fillRect(x0 + (x1 - x0) * i / (N + 1) + 1, sy, Math.max(1, cw - 2), stripH); }
      c.fillStyle = C["ink-3"]; c.font = MONO(10.5); c.textAlign = "left"; c.textBaseline = "middle"; c.fillText("illiquidity", x1 + 6, sy + stripH / 2);
    }
    const last = PL.rounds[PL.rounds.length - 1];
    $("pm-result").innerHTML = PL.settled && last ? (() => {
      const f = (you, att, ta) => `${money(you)} (attacker ${money(att)}, day ${dayOf(ta)}) · ${you - att > 0.005 ? "<b>you did more damage</b>" : you - att < -0.005 ? "the attacker did more" : "a tie"}`;
      return `You exercised on day ${dayOf(last.t)}${p.S[last.t] >= MK.K ? " (out of the money: you get nothing, but the sellers still unwind their hedges)" : ""}.<br>GDA seller lost ${f(last.ya, last.aa, last.ta)}.<br>LSM-trained seller lost ${f(last.yb, last.ab, last.tb)}. All losses before the premium.`;
    })() : "Exercise opens on day 3. If you never exercise, the option is exercised at expiry.";
    // --- seller cards ---
    const card = (id, ro, att, col) => {
      const i = Math.min(n, N - 1);
      $(`${id}-owes`).textContent = n >= 1 ? money(ro.L[PL.settled ? PL.tau : n]) : "—";
      $(`${id}-delta`).textContent = num(ro.delta[i]);
      $(`${id}-cost`).textContent = money(n < N ? ro.C[n + 1] : ro.C[N]);
      const kk = canvas(`cv-${id}`); if (!kk) return;
      const { c, w, h } = kk, x0 = 36, x1 = w - 8, y0 = 8, y1 = h - 18;
      const upto = PL.settled ? N : Math.max(1, n);
      let lo = 0, hi = 5;
      for (let j = 1; j <= upto; j++) { lo = Math.min(lo, ro.L[j]); hi = Math.max(hi, ro.L[j]); }
      hi *= 1.1;
      const X = (j) => x0 + (x1 - x0) * (j - 1) / (N - 1), Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
      frame(c, x0, x1, y0, y1, X, Y, { yt: ticks(lo, hi, 3), yf: (v) => "$" + v, xt: [] });
      if (PL.settled) line(c, Array.from({ length: N }, (_, j) => [X(j + 1), Y(ro.L[j + 1])]), C["ink-3"], 1.2, [2, 3]);
      line(c, Array.from({ length: PL.settled ? PL.tau : n }, (_, j) => [X(j + 1), Y(ro.L[j + 1])]), col, 2);
      if (PL.settled) {
        diamond(c, X(att.tau), Y(ro.L[att.tau]), col);
        dot(c, X(PL.tau), Y(ro.L[PL.tau]), 4, C.holder);
      } else if (n >= 1) dot(c, X(n), Y(ro.L[n]), 3.5, col);
    };
    card("pA", A, PL.attA, C.seller);
    card("pB", B, PL.attB, C.third);
    const S9 = SETS["0.9"];
    $("pA-note").innerHTML = `Its certificate: no exercise rule, yours included, can push its tail loss (CVaR 90%) above <b>${money(S9.duals.gda.D)}</b> over many rounds.`;
    $("pB-note").innerHTML = `It rates its own tail risk at <b>${money(S9.matrix["lsm|lsm_rule"].cvar)}</b>, assuming you follow the textbook exercise rule. Push it above that.`;
    renderPlayScore();
  }

  function renderPlayScore() {
    const R = PL.rounds, n = R.length, S9 = SETS["0.9"];
    const col = (f) => R.map(f), mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
    const set = (id, v) => { $(id).textContent = v; };
    $("ps-sub").textContent = n ? `${n} round${n === 1 ? "" : "s"} played · CVaR at 90%` : "No rounds yet · play a round to start scoring";
    const ya = col((r) => r.ya), aa = col((r) => r.aa), yb = col((r) => r.yb), ab = col((r) => r.ab);
    set("ps-ya", n ? money(mean(ya)) : "—"); set("ps-yb", n ? money(mean(yb)) : "—");
    set("ps-aa", n ? money(mean(aa)) : "—"); set("ps-ab", n ? money(mean(ab)) : "—");
    const wins = (y, a) => y.filter((v, i) => v - a[i] > 0.005).length;
    set("ps-wa", n ? `${wins(ya, aa)} of ${n}` : "—"); set("ps-wb", n ? `${wins(yb, ab)} of ${n}` : "—");
    const tail = (x) => (n >= 10 ? money(Engine.cvar(x, 0.9)) : `after 10 rounds`);
    set("ps-ta", tail(ya)); set("ps-tb", tail(yb)); set("ps-tta", tail(aa)); set("ps-ttb", tail(ab));
    set("ps-oa", money(S9.matrix["gda|advA"].cvar)); set("ps-ob", n >= 10 ? `${money(Engine.cvar(col((r) => r.ob), 0.9))} on your paths` : money(S9.matrix["lsm|lsm_rule"].cvar));
    set("ps-ca", money(S9.duals.gda.D)); set("ps-cb", money(S9.duals.lsm.D));
    // per-round differences, last 30 rounds
    const k = canvas("cv-pscore");
    if (k) {
      const { c, w, h } = k, x0 = 40, x1 = w - 8, y0 = 8, y1 = h - 18, rr = R.slice(-30);
      let m = 1; for (const r of rr) m = Math.max(m, Math.abs(r.ya - r.aa), Math.abs(r.yb - r.ab));
      m = Math.ceil(m);
      const Y = (v) => y0 + (y1 - y0) * (m - v) / (2 * m), slot = (x1 - x0) / 30, bw = Math.max(2, Math.min(9, slot / 2 - 2));
      frame(c, x0, x1, y0, y1, (v) => v, Y, { yt: [-m, 0, m], yf: (v) => (v > 0 ? "+$" : v < 0 ? "−$" : "$") + Math.abs(v), xt: [] });
      line(c, [[x0, Y(0)], [x1, Y(0)]], C["ink-3"], 1);
      rr.forEach((r, i) => {
        const xc = x0 + slot * (i + 0.5);
        for (const [d, colr, off] of [[r.ya - r.aa, C.seller, -1], [r.yb - r.ab, C.third, 1]]) {
          const xa = xc + (off < 0 ? -bw - 1 : 1), ya2 = Y(Math.max(0, d)), yb2 = Y(Math.min(0, d));
          c.fillStyle = colr; c.fillRect(xa, ya2, bw, Math.max(1, yb2 - ya2));
        }
      });
      if (!rr.length) { c.fillStyle = C["ink-3"]; c.font = FONT(12); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("Bars appear here after each round: above zero, you out-damaged the attacker.", (x0 + x1) / 2, (y0 + y1) / 2); }
    }
    let note = "Above zero means you made that seller lose more than its trained attacker did on the same path.";
    if (n >= 10) {
      const tb = Engine.cvar(yb, 0.9), ta = Engine.cvar(ya, 0.9), own = Engine.cvar(col((r) => r.ob), 0.9), ceil = S9.duals.gda.D;
      note += tb > own ? ` Your tail against the LSM-trained seller, ${money(tb)}, is above the ${money(own)} it expected on the same paths: on your rounds, it underpriced its risk.`
        : ` Your tail against the LSM-trained seller, ${money(tb)}, is still below the ${money(own)} it expected on the same paths.`;
      if (ta > ceil) note += ` Against GDA you are above the ${money(ceil)} ceiling, which over ${n} rounds is luck: the certificate holds on average over many rounds. Try your idea in the rule builder.`;
    } else note += " Tail numbers need at least 10 rounds and stay noisy until about 50.";
    $("ps-note").textContent = note;
  }

  // ---------------- rule builder: a simple threshold rule scored on 2,000 fresh paths ----------------
  const RB = { rng: Engine.makeRng(777001), target: 4000, S: [], ell: [], LA: [], LB: [], aa: [], ab: [], ob: [], best: null };
  function rbPrep(budgetMs) {
    if (RB.S.length >= RB.target) return false;
    const t0 = performance.now();
    while (RB.S.length < RB.target && performance.now() - t0 < budgetMs) {
      const p = E.simulate(RB.rng), A = E.rollout("gda", p), B = E.rollout("lsm", p);
      RB.S.push(Float32Array.from(p.S)); RB.ell.push(Float32Array.from(p.ell)); RB.LA.push(A.L); RB.LB.push(B.L);
      RB.aa.push(E.playRound("gda", "advA", p).loss); RB.ab.push(E.playRound("lsm", "advB", p).loss);
      RB.ob.push(E.playRound("lsm", "lsm_rule", p).loss);          // what the LSM-trained seller expects on these paths
    }
    return true;
  }
  const ruleNow = () => ({ x: +$("r-x").value, y: +$("r-y").value, d: +$("r-d").value, itm: $("r-itm").checked });
  function ruleLosses(rule, Ls, from = 0, to = RB.S.length) {
    const out = new Float64Array(to - from), dmin = Math.ceil(rule.d / DAYS * N), ymax = rule.y >= 10 ? Infinity : rule.y;
    for (let i = from; i < to; i++) {
      const L = Ls[i], Sp = RB.S[i], el = RB.ell[i];
      let t = N;
      for (let n = Math.max(1, dmin); n < N; n++) if (L[n] >= rule.x && el[n] <= ymax && (!rule.itm || Sp[n] < MK.K)) { t = n; break; }
      out[i - from] = L[t];
    }
    return out;
  }
  function bestRule(Ls) {          // grid search on the first half, scored on the second half
    const half = Math.floor(RB.S.length / 2);
    let best = null;
    for (let x = 0; x <= 30; x += 0.5) for (const y of [1.25, 1.5, 2, 3, 4, 6, 10]) for (const d of [0, 40, 80]) for (const itm of [false, true]) {
      const r = { x, y, d, itm }, v = Engine.cvar(Array.from(ruleLosses(r, Ls, 0, half)), 0.9);
      if (!best || v > best.v) best = { r, v };
    }
    const out = Engine.cvar(Array.from(ruleLosses(best.r, Ls, half)), 0.9);
    return { ...best, out, half };
  }
  function renderRule() {
    const n = RB.S.length, S9 = SETS["0.9"];
    $("r-x-v").textContent = money(+$("r-x").value);
    $("r-y-v").textContent = +$("r-y").value >= 10 ? "any" : `${(+$("r-y").value).toFixed(2)}×`;
    $("r-d-v").textContent = $("r-d").value;
    $("pr-sub").textContent = n < RB.target ? `Preparing 4,000 fresh paths… ${Math.round(100 * n / RB.target)}%` : `Scored on 4,000 fresh paths · CVaR 90%`;
    $("r-best-a").disabled = $("r-best-b").disabled = n < RB.target;
    const k = canvas("cv-prule"); if (!k) return;
    const { c, w, h } = k;
    if (n < 50) { c.fillStyle = C["ink-3"]; c.font = FONT(12); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("Simulating paths…", w / 2, h / 2); return; }
    const rule = ruleNow();
    const rows = [
      { name: "vs GDA", col: C.seller, you: Engine.cvar(Array.from(ruleLosses(rule, RB.LA)), 0.9), att: Engine.cvar(RB.aa, 0.9), own: null, ceil: S9.duals.gda.D },
      { name: "vs LSM-trained", col: C.third, you: Engine.cvar(Array.from(ruleLosses(rule, RB.LB)), 0.9), att: Engine.cvar(RB.ab, 0.9), own: Engine.cvar(RB.ob, 0.9), ceil: S9.duals.lsm.D },
    ];
    const vals = rows.flatMap((r) => [r.you, r.att, r.ceil, r.own].filter((v) => v != null));
    const lo = Math.floor(Math.min(...vals) - 0.5), hi = Math.ceil(Math.max(...vals) + 0.5);
    const x0 = 110, x1 = w - 14, X = (v) => x0 + (x1 - x0) * (v - lo) / (hi - lo);
    frame(c, x0, x1, 0, h - 22, X, () => 0, { yt: [], xt: ticks(lo, hi, w < 420 ? 4 : 7), xf: (v) => "$" + v });
    rows.forEach((r, i) => {
      const y = 22 + i * 54;
      c.fillStyle = C.third; c.globalAlpha = 0.18; c.fillRect(X(r.ceil), y - 16, x1 - X(r.ceil), 32); c.globalAlpha = 1;
      line(c, [[X(lo), y], [X(hi), y]], C.grid, 6);
      if (r.own != null) line(c, [[X(r.own), y - 14], [X(r.own), y + 14]], C["ink-2"], 1.5, [3, 3]);
      line(c, [[X(r.ceil), y - 16], [X(r.ceil), y + 16]], C.third, 1.5);
      dot(c, X(r.att), y, 5, C.ink); dot(c, X(r.you), y, 6, C.holder);
      c.fillStyle = C.ink; c.font = FONT(12.5, 600); c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(r.name, x0 - 10, y);
      c.font = FONT(11.5); c.fillStyle = C["ink-2"]; c.textAlign = "left";
      c.fillText(`your rule ${money(r.you)} · attacker ${money(r.att)}`, x0, y + 22);
    });
    const [ga, lb] = rows;
    let note = `All numbers are on the same 4,000 paths. Against GDA your rule reaches ${money(ga.you)} and its trained attacker ${money(ga.att)}; the certificate says no rule can pass ${money(ga.ceil)} on average. `;
    note += lb.you > lb.own ? `Against the LSM-trained seller it reaches ${money(lb.you)}, above the ${money(lb.own)} that seller expects on these paths: even this simple rule shows it underprices its tail risk.`
      : `Against the LSM-trained seller it reaches ${money(lb.you)}, below the ${money(lb.own)} that seller expects on these paths. Try a different threshold, or press "Find the best rule".`;
    if (RB.best) note += ` Best rule found ${RB.best.which}: owes at least ${money(RB.best.r.x)}, illiquidity at most ${RB.best.r.y >= 10 ? "any" : RB.best.r.y + "×"}, from day ${RB.best.r.d}${RB.best.r.itm ? ", in the money" : ""}. Fitted on ${RB.best.half.toLocaleString()} paths, it scored ${money(RB.best.out)} on the other ${RB.best.half.toLocaleString()}.`;
    $("pr-note").textContent = note;
  }
  function applyBest(which) {
    if (RB.S.length < RB.target) return;
    const b = bestRule(which === "a" ? RB.LA : RB.LB);
    RB.best = { ...b, which: which === "a" ? "against GDA" : "against the LSM-trained seller" };
    $("r-x").value = b.r.x; $("r-y").value = b.r.y; $("r-d").value = b.r.d; $("r-itm").checked = b.r.itm;
    S.dirty = true;
  }
  function wirePlay() {
    $("pl-start").addEventListener("click", plToggle);
    $("pl-ex").addEventListener("click", plExercise);
    $("pl-next").addEventListener("click", () => plNew(true));
    $("pl-speed").addEventListener("click", () => { PL.speed = PL.speed === 1 ? 2 : PL.speed === 2 ? 0.5 : 1; $("pl-speed").firstChild.textContent = `Speed ${PL.speed}× `; });
    for (const id of ["r-x", "r-y", "r-d", "r-itm"]) $(id).addEventListener("input", () => { RB.best = null; S.dirty = true; });
    $("r-best-a").addEventListener("click", () => applyBest("a"));
    $("r-best-b").addEventListener("click", () => applyBest("b"));
  }

  // ---------------- chips, modes, stepper ----------------
  function buildChips() {
    document.querySelectorAll(".chips[data-group]").forEach((el) => {
      const g = el.dataset.group, list = g === "hedge" ? HEDGES : g === "stop" ? STOPS : SETTLES;
      el.setAttribute("role", "group");
      el.setAttribute("aria-label", g === "hedge" ? "Seller's hedge" : g === "stop" ? "Holder's strategy" : "Settlement");
      el.innerHTML = list.map((x) => `<button class="chip" data-v="${x.k}" aria-pressed="false">${x.name}</button>`).join("");
      el.addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        S[g] = b.dataset.v; syncChips(); replay();
      });
    });
    syncChips();
  }
  function syncChips() {
    const avail = { hedge: SET().h_keys, stop: SET().s_keys };
    document.querySelectorAll(".chips[data-group]").forEach((el) => {
      const g = el.dataset.group;
      el.querySelectorAll("button").forEach((b) => {
        b.setAttribute("aria-pressed", String(b.dataset.v === S[g]));
        if (avail[g]) b.hidden = !avail[g].includes(b.dataset.v);
      });
    });
    document.querySelectorAll("#alphabar .chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.a === S.alpha)));
    markMatrix();
  }
  // ---------------- tail level (alpha) ----------------
  function alphaLabel(a) { return +a === 0 ? "Mean (α 0)" : `CVaR ${Math.round(+a * 100)}%`; }
  function buildAlphaBar() {
    $("alpha-chips").innerHTML = LEVELS.map((a) => `<button class="chip" data-a="${a}" aria-pressed="false">${alphaLabel(a)}</button>`).join("");
    $("alpha-chips").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setAlpha(b.dataset.a); });
  }
  function setAlpha(a) {
    if (!SETS[a]) return;
    S.alpha = a;
    if (!SET().h_keys.includes(S.hedge)) S.hedge = "gda";
    if (!SET().s_keys.includes(S.stop)) S.stop = "advA";
    const st = SET(), it = st.iters || {};
    $("alpha-desc").innerHTML = (+a === 0
      ? "α = 0 scores the plain average loss. Since the stock's expected return is the risk-free rate, hedging adds trading costs but no expected gain, so both trained hedges hold almost no shares."
      : `${alphaLabel(a)} is the average of the seller's worst ${Math.round((1 - +a) * 100)}% of losses.`) +
      ` Every model in this view was trained for this level: GDA hedge and attacker (${(it.gda || 0).toLocaleString()} iterations${st.restarts ? ", with attacker restarts" : ""}), LSM-trained hedge (${(it.lsm || 0).toLocaleString()})` +
      (a === "0.9" ? ", maturity-trained hedge, and an attacker against each." : ", and an attacker against the LSM-trained hedge. The maturity-trained hedge exists only at 90%.");
    buildMatrix(); syncChips();
    if (S.path) replay(); else S.dirty = true;
  }
  const PHASES = {
    game: ["Simulate market", "Seller hedges · holder watches", "Holder exercises", "Score the round"],
    seller: ["Simulate market", "Hedge each day", "Option exercised", "Score the tail risk"],
    holder: ["Simulate market", "Decide each day", "Exercise", "Average into the price"],
  };
  const SHOW = {
    game: ["c-seller", "c-market", "c-holder", "c-matrix", "c-duel", "c-score", "c-train"],
    play: ["c-pA", "c-pmarket", "c-pB", "c-pscore", "c-prule"],
    seller: ["c-market", "c-hedge", "c-score", "c-spolicy"],
    holder: ["c-market", "c-decide", "c-price", "c-hpolicy"],
  };
  const ALL = ["c-pA", "c-pmarket", "c-pB", "c-pscore", "c-prule", "c-seller", "c-market", "c-holder", "c-matrix", "c-duel", "c-score", "c-train", "c-hedge", "c-spolicy", "c-decide", "c-price", "c-hpolicy"];
  let typeset = null;
  function showMath() {
    if (typeset) return;
    const run = () => window.MathJax.typesetPromise([$("math")]).catch(() => {});
    if (window.MathJax && window.MathJax.typesetPromise) typeset = run();
    else typeset = new Promise((res) => {
      const t = setInterval(() => { if (window.MathJax && window.MathJax.typesetPromise) { clearInterval(t); run().then(res); } }, 100);
      setTimeout(() => { clearInterval(t); typeset = null; res(); }, 15000);
    });
  }
  function setMode(m) {
    const was = S.mode;
    S.mode = m;
    const math = m === "math";
    $("math").hidden = !math;
    $("board").hidden = math;
    document.querySelector(".stepper").hidden = math || m === "play";
    $("playbar").hidden = m !== "play";
    if (m === "play" && !PL.path) plNew(false);
    document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
    try { localStorage.setItem("svh-mode", m); } catch (e) { /* storage unavailable */ }
    $("alphabar").hidden = !(m === "game" || m === "seller");
    if (math) { showMath(); S.dirty = true; return; }
    $("board").className = "board mode-" + m;
    for (const id of ALL) $(id).hidden = !SHOW[m].includes(id);
    document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
    $("score-against").hidden = m !== "seller";
    if (m !== "seller") $("sc-train").hidden = true;
    // order the seller-mode cards: market, hedge, risk, policy
    const order = { seller: { "c-hedge": 2, "c-score": 3, "c-spolicy": 4 } }[m] || {};
    for (const id of ALL) $(id).style.order = order[id] || "";
    $("phases").innerHTML = (PHASES[m] || []).map((t) => `<div class="phase"><span class="pdot"></span><span class="label">${t}</span></div>`).join("");
    try { localStorage.setItem("svh-mode", m); } catch (e) { /* storage unavailable */ }
    if (was !== m && (was === "holder" || m === "holder")) { S.n = 0; S.settled = false; }
    S.dirty = true;
  }
  function renderStepper(now) {
    const since = now - S.settleT;
    const ph = !S.settled ? (S.n === 0 ? 0 : 1) : since < 1100 / S.speed ? 2 : 3;
    $("phases").querySelectorAll(".phase").forEach((el, i) => { el.classList.toggle("on", i === ph); el.classList.toggle("done", i < ph); });
    if (!S.batching) $("status").textContent = `Round ${S.rounds + (S.settled ? 0 : 1)} · day ${dayOf(S.n)} of ${DAYS}`;
    $("b-play-t").textContent = S.playing ? "Pause" : "Play";
    $("b-batch-t").textContent = S.mode === "holder" ? "Play 2,000 rounds" : S.mode === "game" ? "Play 1,000 rounds" : "Play 500 rounds";
  }

  // ---------------- loop ----------------
  let last = performance.now(), acc = 0, lastTrain = 0;
  function loop(now) {
    const dt = Math.min(100, now - last); last = now;
    plTick(dt);
    if (S.mode === "play" && rbPrep(14) && (now - lastWarm > 300 || RB.S.length >= RB.target)) { lastWarm = now; S.dirty = true; }
    if (S.playing && !S.batching && S.mode !== "math" && S.mode !== "play") {
      if (!S.settled) {
        acc += dt;
        const stepMs = 230 / S.speed;
        while (acc >= stepMs && !S.settled) { acc -= stepMs; advance(); }
      } else if (now - S.settleT > 2600 / S.speed) { newRound(); acc = 0; }
    }
    if (S.mode === "game" && now - lastTrain > 60) { lastTrain = now; renderTrain(now); }
    warmUp(now);
    if (S.dirty) { S.dirty = false; render(now); }
    renderStepper(now);
    requestAnimationFrame(loop);
  }
  function render(now) {
    if (S.mode !== "play") renderMarket();
    if (S.mode === "game") { renderSeller(); renderHolder(); renderScore(); renderTrain(now); renderDuel(); }
    if (S.mode === "seller") { renderHedge(); renderScore(); renderSPolicy(); }
    if (S.mode === "holder") { renderDecide(); renderPrice(); renderHPolicy(); }
    if (S.mode === "play") { renderPlay(); renderRule(); }
  }

  // ---------------- wiring ----------------
  function wire() {
    $("mx").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      S.hedge = b.dataset.h; S.stop = b.dataset.s; syncChips(); replay();
    });
    document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    $("b-play").addEventListener("click", () => { S.playing = !S.playing; S.dirty = true; });
    $("b-step").addEventListener("click", stepOnce);
    $("b-new").addEventListener("click", () => { newRound(); });
    $("b-batch").addEventListener("click", () => batch(batchSize()));
    $("b-speed").addEventListener("click", cycleSpeed);
    ["m-chart", "sc-chart", "sp-chart", "du-chart"].forEach(hoverable);
    window.addEventListener("resize", () => { S.dirty = true; });
    window.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (S.mode === "math" && !["1", "2", "3", "4", "5"].includes(k)) return;
      if (e.target && e.target.tagName === "INPUT" && k !== "e") return;
      if (k === "5") { setMode("play"); return; }
      if (S.mode === "play") {
        if (k === " ") { e.preventDefault(); plToggle(); }
        else if (k === "e" || k === "enter") { e.preventDefault(); plExercise(); }
        else if (k === "n") plNew(true);
        else if (k === "arrowright") { e.preventDefault(); PL.running = false; plStep(); }
        else if (k === "s") $("pl-speed").click();
        else if (k === "1") setMode("game"); else if (k === "2") setMode("seller"); else if (k === "3") setMode("holder"); else if (k === "4") setMode("math");
        return;
      }
      if (k === " ") { e.preventDefault(); S.playing = !S.playing; S.dirty = true; }
      else if (k === "arrowright") { e.preventDefault(); stepOnce(); }
      else if (k === "n") newRound();
      else if (k === "b") batch(batchSize());
      else if (k === "s") cycleSpeed();
      else if (k === "1") setMode("game"); else if (k === "2") setMode("seller"); else if (k === "3") setMode("holder"); else if (k === "4") setMode("math");
    });
  }
  function fillMath() {
    const S9 = SETS["0.9"], M = S9.matrix, set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    const rowMax = (h) => Math.max(...S9.s_keys.map((s) => M[`${h}|${s}`].cvar));
    const own = M["lsm|lsm_rule"].cvar, wl = rowMax("lsm"), dl = S9.duals.lsm.D, dg = S9.duals.gda.D;
    const pct = `${(100 * (wl / own - 1)).toFixed(1)}–${(100 * (dl / own - 1)).toFixed(1)}%`;
    set("f-saddle", M["gda|advA"].cvar.toFixed(2)); set("f-saddle2", M["gda|advA"].cvar.toFixed(2));
    set("f-wg", rowMax("gda").toFixed(2)); set("f-dg", dg.toFixed(2)); set("f-dg2", dg.toFixed(2)); set("f-dg3", dg.toFixed(2));
    set("f-wl", wl.toFixed(2)); set("f-wl2", wl.toFixed(2)); set("f-dl", dl.toFixed(2)); set("f-dl2", dl.toFixed(2));
    set("f-own", own.toFixed(2)); set("f-pct", pct); set("f-pct2", pct);
    const br = (k) => `[${DATA.pricing[k].lower.toFixed(3)}, ${DATA.pricing[k].upper.toFixed(3)}]`;
    set("f-pc", br("cash_lam1")); set("f-pp", br("phys_lam1")); set("f-pl", br("phys_psi1"));
  }
  // keep filling the head-to-head in the background (a few ms per frame) until 2,000 rounds of this matchup
  let lastWarm = 0;
  function warmUp(now) {
    if (S.mode !== "game" || S.batching || document.hidden || !S.path) return;
    const have = (S.duel[duelKey()] || []).length;
    if (have >= 2000) return;
    const t0 = performance.now();
    while (performance.now() - t0 < 6) { const p = E.simulate(S.rng); recordGame(play(S.hedge, S.stop, p), play(shadowKind(), S.stop, p)); }
    if (now - lastWarm > 400) { lastWarm = now; S.dirty = true; }
  }
  const batchSize = () => (S.mode === "holder" ? 2000 : S.mode === "game" ? 1000 : 500);
  function stepOnce() { S.playing = false; if (S.settled) newRound(); else advance(); }
  function cycleSpeed() { S.speed = S.speed === 1 ? 2 : S.speed === 2 ? 4 : 1; $("b-speed").firstChild.textContent = `Speed ${S.speed}× `; }

  function start() {
    const mk = DATA.market, lq = DATA.liquidity;
    $("foot-model").textContent = `Model: Merton jumps (${mk.lam} per year, mean log-jump ${mk.m}, jump volatility ${mk.delta * 100}%), volatility ${mk.sigma * 100}%, rate ${mk.r * 100}%, 6 months, ${mk.N} exercise dates. Liquidity: spreads and impact rise after sharp falls and decay over about 9 trading days; position ${lq.psi * 100}% of daily volume. The seller's loss includes unwinding the hedge at exercise. Matrix and histogram outlines: ${DATA.test_paths.toLocaleString()} fixed test paths.`;
    buildAlphaBar(); buildMatrix(); buildChips(); wire(); wirePlay(); fillMath(); setAlpha("0.9");
    document.querySelectorAll(".see").forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.goto); window.scrollTo(0, 0); }));
    let m = "game"; try { m = localStorage.getItem("svh-mode") || "game"; } catch (e) { /* storage unavailable */ }
    setMode(["game", "seller", "holder", "math", "play"].includes(m) ? m : "game");
    // a realistic opening state: rounds already played, and a path part-way through
    for (let i = 0; i < 200; i++) { const p = E.simulate(S.rng); recordGame(play(S.hedge, S.stop, p), play(shadowKind(), S.stop, p)); S.rounds++; }
    for (const st of SETTLES) for (let i = 0; i < 500; i++) { const p = E.simulate(S.rng); recordHolder(E.holderRound(st.k, p), p, st.k); }
    newRound();
    while (!S.settled && S.n < Math.min(28, tau() - 1)) advance();
    requestAnimationFrame((t) => { last = t; loop(t); });
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(start, start); else start();
})();
