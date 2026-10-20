// Parity test: the JavaScript engine reproduces the Python simulator, networks and rollouts on shared shocks.
// Usage: node dashboard/test_parity.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const Engine = createRequire(import.meta.url)(join(here, "engine.js"));
const data = JSON.parse(readFileSync(join(here, "dashboard_data.json"), "utf8"));
const E = Engine.create(data);
const P = data.parity, N = E.N;
let worst = {}, fails = 0;
const check = (name, a, b, tol) => {
  const d = Math.abs(a - b);
  worst[name] = Math.max(worst[name] ?? 0, d);
  if (!(d <= tol)) { if (fails++ < 10) console.log(`FAIL ${name}: js ${a} py ${b}`); }
};
for (let i = 0; i < P.S.length; i++) {
  const sh = Object.fromEntries(["Z", "U", "Zj", "eps", "eps_v"].map((k) => [k, P.shocks[k][i]]));
  const p = E.simulate(null, sh);
  for (let n = 0; n <= N; n++) {
    check("S", p.S[n], P.S[i][n], 1e-9 * P.S[i][n]);
    check("ell", p.ell[n], P.ell[i][n], 1e-9);
    check("vol", p.vol[n], P.vol[i][n], 1e-9);
    check("sig", p.sig[n], P.sig[i][n], 1e-9);
  }
  const ro = E.rollout("gda", p);
  const rd = E.rollout("delta", p);
  for (let n = 1; n <= N; n++) {
    check("delta_gda", ro.delta[n - 1], P.delta[i][n - 1], 2e-4);
    check("L_gda", ro.L[n], P.L[i][n - 1], 2e-3);
    check("logit_advA", E.stopLogit("advA", p, ro, n), P.logitA[i][n - 1], 2e-3);
    check("delta_merton", rd.delta[n - 1], P.delta_merton[i][n - 1], 1e-5);
    check("L_merton", rd.L[n], P.L_delta[i][n - 1], 2e-3);
    check("lsm_stop", E.stopLogit("lsm_rule", p, ro, n) > 0 ? 1 : 0, P.lsm_stop[i][n - 1], 0);
    for (const name of ["cash_lam1", "phys_lam1"]) {
      const ev = E.exerciseValue(name, p, n);
      check(`Y_${name}`, ev.Y, P.pricing[name].Y[i][n - 1], 1e-9);
      if (n < N) check(`C_${name}`, E.continuation(name, p, n, ev.Y), P.pricing[name].C[i][n - 1], 1e-4);
    }
  }
}
for (const [k, v] of Object.entries(worst)) console.log(`${k.padEnd(16)} max |js - py| = ${v.toExponential(2)}`);
// speed: rounds per second for the default matchup
const rng = Engine.makeRng(7);
const t0 = performance.now();
let n = 0;
while (performance.now() - t0 < 1000) { E.playRound("gda", "advA", E.simulate(rng)); n++; }
console.log(`${n} game rounds per second (gda vs advA)`);
console.log(fails ? `${fails} FAILURES` : "ALL PARITY CHECKS PASS");
process.exit(fails ? 1 : 0);
