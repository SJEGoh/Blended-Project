// Path demo engine: Merton (+ liquidity) path simulation, hedger / adversary networks, LSM rule.
// Mirrors sim.py, losses.py and models.py (checked against frontend/verify_demo.py by
// frontend/test_engine.js). Works in the browser (window.DemoEngine) and in node (module.exports).
(function (root) {
  "use strict";

  // ---------- seeded RNG ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    const u = mulberry32(seed);
    let spare = null;
    return {
      uniform: () => { let x; do { x = u(); } while (x === 0); return x; },
      normal() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let a, b; do { a = u(); } while (a === 0); b = u();
        const r = Math.sqrt(-2 * Math.log(a)), th = 2 * Math.PI * b;
        spare = r * Math.sin(th);
        return r * Math.cos(th);
      },
    };
  }

  function poissonInv(U, rate) {
    if (rate <= 0) return 0;
    const kmax = Math.ceil(rate + 12 * Math.sqrt(rate) + 12);
    let pmf = Math.exp(-rate), cdf = pmf, n = U > cdf ? 1 : 0;
    for (let k = 1; k <= kmax; k++) { pmf *= rate / k; cdf += pmf; if (U > cdf) n++; }
    return n;
  }

  // ---------- simulation (sim.simulate / sim.liquidity_state, one path) ----------
  // Returns { logS[N+1], jumps[N] (log jump size per step, 0 if none), extras|null }
  function simulatePath(mk, liq, seed) {
    const rng = makeRng(seed);
    const N = mk.N, dt = mk.T / N, kappa = Math.exp(mk.m + 0.5 * mk.delta * mk.delta) - 1;
    const Z = [], U = [], Zj = [];
    for (let n = 0; n < N; n++) Z.push(rng.normal());
    for (let n = 0; n < N; n++) U.push(rng.uniform());
    for (let n = 0; n < N; n++) Zj.push(rng.normal());
    const drift = (mk.r - mk.lam * kappa - 0.5 * mk.sigma * mk.sigma) * dt;
    const logS = [Math.log(mk.S0)], jumps = [];
    for (let n = 0; n < N; n++) {
      const nj = poissonInv(U[n], mk.lam * dt);
      const j = nj * mk.m + mk.delta * Math.sqrt(nj) * Zj[n];
      jumps.push(nj > 0 ? j : 0);
      logS.push(logS[n] + drift + mk.sigma * Math.sqrt(dt) * Z[n] + j);
    }
    const extras = liq ? liquidityState(mk, liq, logS, rng) : null;
    return { logS, jumps, extras };
  }

  function liquidityState(mk, lq, logS, rng) {
    const N = logS.length - 1, dt = mk.T / mk.N, sd = mk.sigma * Math.sqrt(dt);
    const eps = [], epsV = [];
    for (let n = 0; n < N; n++) eps.push(rng.normal());
    for (let n = 0; n < N; n++) epsV.push(rng.normal());
    const x = [0], s2 = [mk.sigma * mk.sigma], logv = [0], xmax = Math.log(lq.ell_max);
    for (let n = 0; n < N; n++) {
      const r = logS[n + 1] - logS[n], drive = Math.max(-r - lq.h_mult * sd, 0);
      x.push(Math.min(x[n] - lq.kappa * x[n] * dt + lq.eta * Math.sqrt(dt) * eps[n] + lq.beta_down * drive, xmax));
      s2.push((1 - lq.ewma_w) * s2[n] + lq.ewma_w * r * r / dt);
      const zabs = Math.min(Math.abs(r / sd), lq.zcap);
      logv.push(lq.gamma_v * (zabs - Math.sqrt(2 / Math.PI)) + lq.zeta * epsV[n] - 0.5 * lq.zeta * lq.zeta);
    }
    return { ell: x.map(Math.exp), vol: logv.map(Math.exp), sig: s2.map(Math.sqrt) };
  }

  // ---------- networks ----------
  function mlp(layers, x) {
    for (let i = 0; i < layers.length; i++) {
      const { W, b } = layers[i], y = new Array(W.length);
      for (let r = 0; r < W.length; r++) {
        let s = b[r];
        const w = W[r];
        for (let c = 0; c < w.length; c++) s += w[c] * x[c];
        y[r] = i < layers.length - 1 ? s / (1 + Math.exp(-s)) : s;   // SiLU on hidden layers
      }
      x = y;
    }
    return x[0];
  }

  function liqCost(q, Sd, ell, vol, sig, lq, sigma) {
    const ref = lq.sigma_ref || sigma, sr = Math.min(sig / ref, lq.sig_ratio_max);
    const hs = lq.cost_scale * lq.s0 * ell * sr;
    const imp = lq.cost_scale * lq.Y * ell * (sr * ref / Math.sqrt(252)) * Math.sqrt(lq.psi / vol);
    return (hs * q + imp * Math.pow(Math.max(q, 0), 1.5)) * Sd;
  }
  function liqFeats(ex, n, lq, sigma) {
    const ref = lq.sigma_ref || sigma;
    return [Math.log(ex.ell[n]), Math.log(ex.vol[n]), Math.log(ex.sig[n] / ref)];
  }

  // ---------- one hedger + its adversary along one path (losses.hedge_rollout / stopper_logits) ----------
  // Arrays over dates 1..N (index 0 = t_1): L (seller loss if exercised then), D (delta held into that
  // date), G, C, U, Zd, logit. Also deltas[] at t_0..t_{N-1}.
  function rollout(sc, hedger, stopper, path) {
    const mk = sc.market, liq = sc.liquidity, c = sc.costs.c || 0, unwind = sc.costs.unwind || "none";
    const N = mk.N, T = mk.T, K = mk.K, ex = path.extras;
    const xs = mk.sigma * Math.sqrt(T), ps = K * mk.sigma * Math.sqrt(T);
    const t = [], S = [], Sd = [], Zd = [];
    for (let n = 0; n <= N; n++) {
      t.push(T * n / N); S.push(Math.exp(path.logS[n]));
      const disc = Math.exp(-mk.r * t[n]);
      Sd.push(S[n] * disc); Zd.push(Math.max(K - S[n], 0) * disc);
    }
    let dp = 0, G = 0, C = 0;
    const out = { L: [], D: [], G: [], C: [], U: [], Zd: [], logit: [], tradeCost: [] };
    for (let n = 0; n < N; n++) {
      const f = [t[n] / T, Math.log(S[n] / K) / xs, dp];
      if (liq) f.push(...liqFeats(ex, n, liq, mk.sigma));
      const d = mlp(hedger, f);
      let tc = c * Math.abs(d - dp) * Sd[n];
      if (liq) tc += liqCost(Math.abs(d - dp), Sd[n], ex.ell[n], ex.vol[n], ex.sig[n], liq, mk.sigma);
      C += tc;
      G += d * (Sd[n + 1] - Sd[n]);
      let Un = 0;
      if (unwind === "cash" && (c > 0 || liq)) {
        Un = c * Math.abs(d) * Sd[n + 1];
        if (liq) Un += liqCost(Math.abs(d), Sd[n + 1], ex.ell[n + 1], ex.vol[n + 1], ex.sig[n + 1], liq, mk.sigma);
      }
      const L = Zd[n + 1] - G + C + Un;
      out.D.push(d); out.G.push(G); out.C.push(C); out.U.push(Un); out.Zd.push(Zd[n + 1]); out.L.push(L); out.tradeCost.push(tc);
      // adversary's view at t_{n+1}
      const g = [t[n + 1] / T, Math.log(S[n + 1] / K) / xs, d, L / ps, S[n + 1] < K ? 1 : 0];
      if (liq) g.push(...liqFeats(ex, n + 1, liq, mk.sigma));
      out.logit.push(stopper ? mlp(stopper, g) : NaN);
      dp = d;
    }
    out.tauAdv = firstTrue(out.logit.map(v => v > 0));    // 0-based over dates 1..N
    return out;
  }

  function firstTrue(mask) {
    for (let i = 0; i < mask.length - 1; i++) if (mask[i]) return i;
    return mask.length - 1;
  }

  // ---------- LSM rule (benchmarks.LSMRule) ----------
  function lsmContinuation(rule, n, S) {
    const x = S / rule.K - 1, co = rule.coefs[n];
    let v = 0, p = 1;
    for (let k = 0; k <= rule.deg; k++) { v += co[k] * p; p *= x; }
    return v;
  }
  function lsmExercise(rule, n, S) {
    const N = rule.t.length - 1;
    if (n >= N) return true;
    if (n <= 0) return false;
    const pay = Math.exp(-rule.r * rule.t[n]) * Math.max(rule.K - S, 0);
    return pay > 0 && pay > lsmContinuation(rule, n, S);
  }
  function lsmStop(rule, logS) {   // 0-based over dates 1..N
    const N = logS.length - 1, mask = [];
    for (let n = 1; n <= N; n++) mask.push(lsmExercise(rule, n, Math.exp(logS[n])));
    return firstTrue(mask);
  }
  // Largest S in [lo, K] where the LSM rule exercises at date n (NaN if none): the exercise boundary.
  function lsmBoundary(rule, n, lo) {
    let best = NaN;
    for (let i = 0; i <= 400; i++) {
      const S = lo + (rule.K - lo) * i / 400;
      if (lsmExercise(rule, n, S)) best = S;
    }
    return best;
  }

  function cvar(X, alpha) {
    const s = X.slice().sort((a, b) => b - a), k = Math.max(1, Math.ceil((1 - alpha) * s.length - 1e-9));
    let m = 0; for (let i = 0; i < k; i++) m += s[i];
    return { cvar: m / k, var: s[k - 1] };
  }

  const api = { makeRng, simulatePath, rollout, lsmStop, lsmExercise, lsmBoundary, cvar, mlp };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DemoEngine = api;
})(typeof window !== "undefined" ? window : this);
