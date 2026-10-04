// Check demo_engine.js against the numpy reference and the published results.
//   python frontend/verify_demo.py --log2 12 --fixture /tmp/fixture.json
//   node frontend/test_engine.js /tmp/fixture.json [n_paths]
// 1) identical path -> network outputs, losses and exercise dates must match the numpy reference;
// 2) the engine's own simulator over n_paths fresh paths must reproduce the published CVaRs.
const fs = require("fs");
const path = require("path");
const E = require("./demo_engine.js");

global.window = {};
eval(fs.readFileSync(path.join(__dirname, "models.js"), "utf8"));
const M = window.DEMO_MODELS;

let worst = 0, fails = 0;
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
for (const fx of fixture) {
  const sc = M.scenarios.find(s => s.id === fx.id);
  const p = { logS: fx.logS, extras: fx.extras };
  for (const [hk, ak] of [["hedger_gda", "adv_vs_gda"], ["hedger_lsm", "adv_vs_lsm"]]) {
    const ro = E.rollout(sc, sc.nets[hk], sc.nets[ak], p);
    for (const [k, ref] of [["L", fx[hk].L], ["D", fx[hk].D], ["logit", fx[hk].logits]]) {
      const err = Math.max(...ref.map((v, i) => Math.abs(v - ro[k][i])));
      worst = Math.max(worst, err);
      if (err > 1e-6) { fails++; console.log(`MISMATCH ${fx.id} ${hk} ${k}: max abs err ${err}`); }
    }
  }
  if (E.lsmStop(M.lsm_rule, fx.logS) !== fx.tau_lsm) { fails++; console.log(`MISMATCH ${fx.id} LSM stop`); }
}
console.log(`identical-path check: ${fixture.length} scenarios, max abs error ${worst.toExponential(2)}, ${fails} mismatches`);

const nPaths = +(process.argv[3] || 0);
if (nPaths) {
  const pub = {
    liqbase_uwcash: "headline_liqbase_uwcash.json", "c0.005_uwcash": "headline_c0.005_uwcash.json",
    "c0.005": "headline_c0.005.json", frictionless: "headline.json",
  };
  for (const sc of M.scenarios) {
    const H = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "results", pub[sc.id]), "utf8")).hedgers;
    const X = { gda: [], "fixed-lsm": [] };
    for (let i = 0; i < nPaths; i++) {
      const p = E.simulatePath(sc.market, sc.liquidity, 1000 + i);
      for (const [src, hk, ak] of [["gda", "hedger_gda", "adv_vs_gda"], ["fixed-lsm", "hedger_lsm", "adv_vs_lsm"]]) {
        const ro = E.rollout(sc, sc.nets[hk], sc.nets[ak], p);
        X[src].push(ro.L[ro.tauAdv]);
      }
    }
    for (const src of ["gda", "fixed-lsm"]) {
      const v = E.cvar(X[src], 0.9).cvar, ref = H.find(h => h.source === src).evals["fresh adversary"].cvar;
      console.log(`${sc.id.padEnd(15)} ${src.padEnd(10)} vs fresh adversary: engine ${v.toFixed(3)}  published ${ref.toFixed(3)}`);
    }
  }
}
process.exit(fails ? 1 : 0);
