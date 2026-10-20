/* Seller vs Holder engine: Merton jump-diffusion + stochastic liquidity simulator, the trained networks'
   forward passes, the seller's hedged-loss rollout and the holder's exercise rules.
   A JavaScript port of sim.py, losses.py, models.py and pricing.py (checked by test_parity.mjs). */
(function (root) {
  "use strict";

  // ---------- numerics ----------
  function erfc(x) {               // W. J. Cody-style rational approximation (|rel err| < 1.2e-7)
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const normCdf = (x) => 0.5 * erfc(-x / Math.SQRT2);
  function lgamma(n) { let s = 0; for (let k = 2; k <= n; k++) s += Math.log(k); return s; }   // integer n

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
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
      uniform: () => { let x = u(); while (x <= 0) x = u(); return x; },
      normal() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let a = u(); while (a <= 1e-300) a = u();
        const b = u(), rad = Math.sqrt(-2 * Math.log(a));
        spare = rad * Math.sin(2 * Math.PI * b);
        return rad * Math.cos(2 * Math.PI * b);
      },
    };
  }

  function poissonInv(U, rate) {   // same truncation as sim._poisson_inv_cdf
    if (rate <= 0) return 0;
    const kmax = Math.ceil(rate + 12 * Math.sqrt(rate) + 12);
    let pmf = Math.exp(-rate), cdf = pmf, count = U > cdf ? 1 : 0;
    for (let k = 1; k <= kmax; k++) { pmf *= rate / k; cdf += pmf; if (U > cdf) count++; }
    return count;
  }

  // ---------- networks ----------
  function decodeF32(b64) {
    let bytes;
    if (typeof atob === "function") {
      const s = atob(b64); bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else bytes = new Uint8Array(Buffer.from(b64, "base64"));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  function makeMLP(blob) {       // nn.Sequential(Linear, SiLU, Linear, SiLU, Linear)
    const flat = decodeF32(blob.data);
    let off = 0;
    const layers = blob.shapes.map(([out, inp]) => {
      const W = flat.subarray(off, off + out * inp); off += out * inp;
      const b = flat.subarray(off, off + out); off += out;
      return { W, b, out, inp };
    });
    const bufs = layers.map((l) => new Float64Array(l.out));
    return function forward(x) {
      let h = x;
      for (let li = 0; li < layers.length; li++) {
        const { W, b, out, inp } = layers[li], y = bufs[li], last = li === layers.length - 1;
        for (let o = 0; o < out; o++) {
          let s = b[o];
          const row = o * inp;
          for (let i = 0; i < inp; i++) s += W[row + i] * h[i];
          y[o] = last ? s : s / (1 + Math.exp(-s));
        }
        h = y;
      }
      return h[0];
    };
  }

  // ---------- model ----------
  function create(data) {
    const mk = data.market, lq = data.liquidity;
    const N = mk.N, dt = mk.T / N, sd = mk.sigma * Math.sqrt(dt);
    const kappaJ = Math.exp(mk.m + 0.5 * mk.delta * mk.delta) - 1;
    const drift = (mk.r - mk.lam * kappaJ - 0.5 * mk.sigma * mk.sigma) * dt;
    const xScale = mk.sigma * Math.sqrt(mk.T), pnlScale = mk.K * mk.sigma * Math.sqrt(mk.T);
    const sigRef = lq.sigma_ref, t = Array.from({ length: N + 1 }, (_, n) => mk.T * n / N);
    const disc = t.map((tt) => Math.exp(-mk.r * tt));

    const hedgeNets = {}, stopNets = {};
    for (const k in data.hedgers) hedgeNets[k] = makeMLP(data.hedgers[k]);
    for (const k in data.stoppers) stopNets[k] = makeMLP(data.stoppers[k]);
    const priceNets = {};
    for (const k in data.pricing) priceNets[k] = data.pricing[k].nets.map(makeMLP);

    // liquidity cost of trading q >= 0 shares at a discounted price Sd (losses.liquidity_cost)
    function liqCost(q, Sd, ell, vol, sig, psi) {
      const sr = Math.min(sig / sigRef, lq.sig_ratio_max);
      const half = lq.cost_scale * lq.s0 * ell * sr;
      const impact = lq.cost_scale * lq.Y * ell * (sr * sigRef / Math.sqrt(252)) * Math.sqrt((psi ?? lq.psi) / vol);
      return (half * q + impact * Math.pow(Math.max(q, 0), 1.5)) * Sd;
    }

    // one path; `shocks` (optional) = {Z, U, Zj, eps, eps_v} arrays of length N for reproducibility tests
    function simulate(rng, shocks) {
      const logS = new Float64Array(N + 1), x = new Float64Array(N + 1), s2 = new Float64Array(N + 1);
      const logv = new Float64Array(N + 1), Nj = new Int32Array(N), ret = new Float64Array(N);
      logS[0] = Math.log(mk.S0); s2[0] = mk.sigma * mk.sigma;
      const xMax = Math.log(lq.ell_max), rate = mk.lam * dt;
      for (let n = 0; n < N; n++) {
        const Z = shocks ? shocks.Z[n] : rng.normal();
        const U = shocks ? shocks.U[n] : rng.uniform();
        const Zj = shocks ? shocks.Zj[n] : rng.normal();
        const nj = poissonInv(U, rate);
        Nj[n] = nj;
        const incr = drift + sd * Z + nj * mk.m + mk.delta * Math.sqrt(nj) * Zj;
        ret[n] = incr;
        logS[n + 1] = logS[n] + incr;
      }
      for (let n = 0; n < N; n++) {
        const eps = shocks ? shocks.eps[n] : rng.normal();
        const epsV = shocks ? shocks.eps_v[n] : rng.normal();
        const r = ret[n];
        x[n + 1] = Math.min(x[n] - lq.kappa * x[n] * dt + lq.eta * Math.sqrt(dt) * eps
          + lq.beta_down * Math.max(-r - lq.h_mult * sd, 0), xMax);
        s2[n + 1] = (1 - lq.ewma_w) * s2[n] + lq.ewma_w * r * r / dt;
        logv[n + 1] = lq.gamma_v * (Math.min(Math.abs(r / sd), lq.zcap) - Math.sqrt(2 / Math.PI)) + lq.zeta * epsV - 0.5 * lq.zeta * lq.zeta;
      }
      const S = Array.from(logS, Math.exp);
      return {
        S, Sd: S.map((s, n) => s * disc[n]), Zd: S.map((s, n) => Math.max(mk.K - s, 0) * disc[n]),
        ell: Array.from(x, Math.exp), vol: Array.from(logv, Math.exp), sig: Array.from(s2, Math.sqrt), Nj: Array.from(Nj),
      };
    }

    // ---- Merton European put delta (reference hedge) ----
    function mertonTerms(tau) {
      const out = [], lamP = mk.lam * (1 + kappaJ);
      if (mk.lam === 0) return [{ w: 1, r: mk.r, s: mk.sigma }];
      let cum = 0;
      for (let n = 0; n < 200; n++) {
        const w = Math.exp(-lamP * tau + n * Math.log(lamP * tau) - lgamma(n));
        out.push({ w, s: Math.sqrt(mk.sigma ** 2 + n * mk.delta ** 2 / tau), r: mk.r - mk.lam * kappaJ + n * Math.log1p(kappaJ) / tau });
        cum += w;
        if (1 - cum < 1e-14 && n > lamP * tau) break;
      }
      return out;
    }
    const termCache = new Map();
    function mertonDelta(S, tau) {
      tau = Math.max(tau, 1e-12);
      const key = tau.toFixed(10);
      let terms = termCache.get(key);
      if (!terms) { terms = mertonTerms(tau); termCache.set(key, terms); }
      let d = 0;
      for (const { w, r, s } of terms) {
        const d1 = (Math.log(S / mk.K) + (r + 0.5 * s * s) * tau) / (s * Math.sqrt(tau));
        d += w * (normCdf(d1) - 1);
      }
      return d;
    }

    function hedgeAt(kind, p, n, dPrev) {
      if (kind === "none") return 0;
      if (kind === "delta") return mertonDelta(p.S[n], mk.T - t[n]);
      return hedgeNets[kind]([t[n] / mk.T, Math.log(p.S[n] / mk.K) / xScale, dPrev,
        Math.log(p.ell[n]), Math.log(p.vol[n]), Math.log(p.sig[n] / sigRef)]);
    }

    // seller's rollout: delta[n] held over [t_n, t_{n+1}); L[n] (n = 1..N) = loss if exercised at t_n
    function rollout(kind, p) {
      const delta = new Float64Array(N), trade = new Float64Array(N), cost = new Float64Array(N);
      const L = new Float64Array(N + 1), G = new Float64Array(N + 1), C = new Float64Array(N + 1), U = new Float64Array(N + 1);
      let dPrev = 0, g = 0, c = 0;
      for (let n = 0; n < N; n++) {
        const d = hedgeAt(kind, p, n, dPrev);
        const cn = liqCost(Math.abs(d - dPrev), p.Sd[n], p.ell[n], p.vol[n], p.sig[n]);
        c += cn; g += d * (p.Sd[n + 1] - p.Sd[n]);
        delta[n] = d; trade[n] = d - dPrev; cost[n] = cn;
        const m = n + 1;
        U[m] = data.costs.unwind === "cash" ? liqCost(Math.abs(d), p.Sd[m], p.ell[m], p.vol[m], p.sig[m]) : 0;
        G[m] = g; C[m] = c; L[m] = p.Zd[m] - g + c + U[m];
        dPrev = d;
      }
      return { delta, trade, cost, L, G, C, U };
    }

    // holder strategies against the seller: logit at date n (1..N); exercise at first logit > 0, forced at N
    function lsmContinuation(n, S) {
      const c = data.lsm_rule.coefs[n], x = S / mk.K - 1;
      let v = 0, xp = 1;
      for (let i = 0; i < c.length; i++) { v += c[i] * xp; xp *= x; }
      return v;
    }
    function stopLogit(kind, p, ro, n) {
      if (kind === "never") return -1;
      if (kind === "lsm_rule") {
        if (n >= N) return 1;
        const pay = disc[n] * Math.max(mk.K - p.S[n], 0);
        return pay > 0 && pay > lsmContinuation(n, p.S[n]) ? 1 : -1;
      }
      return stopNets[kind]([t[n] / mk.T, Math.log(p.S[n] / mk.K) / xScale, ro.delta[n - 1], ro.L[n] / pnlScale,
        p.S[n] < mk.K ? 1 : 0, Math.log(p.ell[n]), Math.log(p.vol[n]), Math.log(p.sig[n] / sigRef)]);
    }
    function playRound(hKind, sKind, p) {
      const ro = rollout(hKind, p), logits = new Float64Array(N + 1);
      let tau = N;
      for (let n = 1; n <= N; n++) {
        logits[n] = stopLogit(sKind, p, ro, n);
        if (n < N && logits[n] > 0 && tau === N) tau = n;
      }
      return { ro, logits, tau, loss: ro.L[tau] };
    }

    // ---- pricing: holder's own exercise value and continuation value (pricing.py) ----
    function exerciseValue(name, p, n) {
      const P = data.pricing[name], Kd = mk.K * disc[n];
      let kappa = 0;
      if (P.mode === "liquidity") kappa = liqCost(1, p.Sd[n], p.ell[n], p.vol[n], p.sig[n], P.psi);
      return { Y: Math.max(Kd - p.Sd[n] - kappa, 0), kappa };
    }
    function continuation(name, p, n, Y) {     // n = 1..N-1
      const P = data.pricing[name];
      const f = [n / N, Math.log(p.S[n] / mk.K) / xScale, p.S[n] < mk.K ? 1 : 0, Y / pnlScale];
      if (P.n_feat === 7) f.push(Math.log(p.ell[n]), Math.log(p.vol[n]), Math.log(p.sig[n] / sigRef));
      return priceNets[name][n - 1](f) * pnlScale;
    }
    function holderRound(name, p) {
      const Y = new Float64Array(N + 1), C = new Float64Array(N + 1), kap = new Float64Array(N + 1);
      let tau = N;
      for (let n = 1; n <= N; n++) {
        const ev = exerciseValue(name, p, n);
        Y[n] = ev.Y; kap[n] = ev.kappa;
        C[n] = n < N ? continuation(name, p, n, ev.Y) : 0;
        if (n < N && tau === N && ev.Y > 0 && ev.Y >= C[n]) tau = n;
      }
      return { Y, C, kappa: kap, tau, payoff: Y[tau] };
    }
    // exercise boundary at date n for a liquidity level: largest S on a grid where the rule exercises
    function boundary(name, n, ell, grid) {
      let best = NaN;
      for (const S of grid) {
        const p = { S: [], Sd: [], ell: [], vol: [], sig: [] };
        p.S[n] = S; p.Sd[n] = S * disc[n]; p.ell[n] = ell; p.vol[n] = 1; p.sig[n] = mk.sigma;
        const ev = exerciseValue(name, p, n);
        if (ev.Y > 0 && ev.Y >= continuation(name, p, n, ev.Y)) best = isNaN(best) ? S : Math.max(best, S);
      }
      return best;
    }

    return { N, dt, t, disc, market: mk, liq: lq, simulate, rollout, hedgeAt, mertonDelta, stopLogit, playRound,
      exerciseValue, continuation, holderRound, boundary, liqCost, xScale, pnlScale };
  }

  function cvar(xs, alpha) {
    if (!xs.length) return NaN;
    const s = Float64Array.from(xs).sort();
    const k = Math.max(1, Math.min(s.length, Math.ceil((1 - alpha) * s.length - 1e-9)));
    let a = 0; for (let i = s.length - k; i < s.length; i++) a += s[i];
    return a / k;
  }

  const Engine = { create, makeRng, cvar, normCdf };
  root.Engine = Engine;
  if (typeof module !== "undefined" && module.exports) module.exports = Engine;
})(typeof window !== "undefined" ? window : globalThis);
