"""CURRENT_PLAN.md Step 5 -- Guard-bias Algorithm 1 (per-channel independent threshold search).

E_benign = {Qv(clean), Qd(clean), Qv(triggered)}   -- must land BELOW V
E_adv    = {Qd(triggered)}                          -- must land ABOVE V

Per dclbd_guardbias_mechanism (folded into CURRENT_PLAN.md §2): searched independently per
channel (not a joint logistic-regression combination across channels -- that was Phase B v2's
mistake and likely why it didn't generalize). tau starts at 0.95, backs off by 0.05 steps if no
threshold separates well enough, down to a floor.

Run from repo root:
  python3 chain_survival/scripts/guard_bias_search.py
"""
import json
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TAU_FLOOR = 0.60
TAU_STEP = 0.05
STEPS = 400


def search_channel(benign, adv, tau_start=0.95):
    """DcL-BD's Algorithm 1 assumes the trigger pushes activation UP (adv > V > benign) --
    that's their FP-reordering mechanism's direction. Our sweep (sweep_target_value.py) found
    the exploitable direction for this INT8-granularity mechanism is DOWN (adv should be the
    MORE NEGATIVE side). Rather than assume either sign, try both per channel and keep whichever
    achieves the higher tau -- the sign is a property of the channel/mechanism, not universal."""
    lo, hi = min(benign.min(), adv.min()), max(benign.max(), adv.max())
    if hi - lo < 1e-9:
        return None
    candidates = np.linspace(lo, hi, STEPS)
    best = None
    for direction in ("adv_above", "adv_below"):
        tau = tau_start
        while tau >= TAU_FLOOR:
            for V in candidates:
                if direction == "adv_above":
                    p_m, p_c = (benign < V).mean(), (adv > V).mean()
                else:
                    p_m, p_c = (benign > V).mean(), (adv < V).mean()
                if p_m > tau and p_c > tau:
                    cand = {"V": float(V), "tau_achieved": float(tau), "direction": direction,
                            "p_benign_ok": float(p_m), "p_adv_ok": float(p_c)}
                    if best is None or cand["tau_achieved"] > best["tau_achieved"]:
                        best = cand
                    break
            else:
                tau -= TAU_STEP
                continue
            break
    return best


def main():
    d = np.load(os.path.join(RESULTS, "fourgroups_guard.npz"))
    gpu_clean, dla_clean, gpu_trig, dla_trig = d["gpu_clean"], d["dla_clean"], d["gpu_trig"], d["dla_trig"]
    n_channels = gpu_clean.shape[1]
    print(f"[guard-bias] searching {n_channels} channels independently...", flush=True)

    results = []
    for c in range(n_channels):
        benign = np.concatenate([gpu_clean[:, c], dla_clean[:, c], gpu_trig[:, c]])
        adv = dla_trig[:, c]
        r = search_channel(benign, adv)
        if r is not None:
            r["channel"] = c
            results.append(r)

    results.sort(key=lambda r: (-r["tau_achieved"], -min(r["p_benign_ok"], r["p_adv_ok"])))
    print(f"[guard-bias] {len(results)}/{n_channels} channels found a separating threshold "
          f"(tau>={TAU_FLOOR})", flush=True)
    print("[guard-bias] top 10 channels:")
    for r in results[:10]:
        print(f"  ch{r['channel']:4d}  V={r['V']:8.2f}  tau={r['tau_achieved']:.2f}  dir={r['direction']:10s} "
              f"P(benign_ok)={r['p_benign_ok']:.3f}  P(adv_ok)={r['p_adv_ok']:.3f}")

    ch0 = [r for r in results if r["channel"] == 0]
    if ch0:
        print(f"[guard-bias] engineered channel 0 result: {ch0[0]}")
    else:
        print("[guard-bias] engineered channel 0 did NOT find any separating threshold (tau<0.60)")

    json.dump(results, open(os.path.join(RESULTS, "guard_bias_search.json"), "w"), indent=2)
    print(f"wrote {RESULTS}/guard_bias_search.json")


if __name__ == "__main__":
    main()
