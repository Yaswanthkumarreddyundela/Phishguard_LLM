import pickle
import numpy as np

with open("data/models/lgbm_model.pkl", "rb") as f:
    data = pickle.load(f)

model_full       = data["model_full"]
model_structural = data["model_structural"]
names_full       = data["feature_names_full"]
names_struct     = data["feature_names_struct"]

def show_importance(model, names, label):
    try:
        gain  = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
    except AttributeError:
        gain  = model.feature_importances_
        split = model.feature_importances_

    total_gain = gain.sum() or 1
    pairs = sorted(zip(names, gain, split), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'Feature':<42} {'Gain%':>7}  {'Splits':>7}")
    print(f"  {'-'*42} {'-'*7}  {'-'*7}")

    dead = []
    for name, g, s in pairs:
        pct = g / total_gain * 100
        if pct < 0.1 and s == 0:
            dead.append(name)
            continue
        bar = "X" * int(pct / 2)
        print(f"  {name:<42} {pct:>6.1f}%  {s:>7}  {bar}")

    if dead:
        print(f"\n  ZERO IMPORTANCE (never used):")
        for d in dead:
            print(f"    - {d}")

show_importance(model_full,       names_full,   "MODEL FULL (38 features)")
show_importance(model_structural, names_struct, "MODEL STRUCTURAL (37 features)")

gain_full  = dict(zip(names_full,   model_full.booster_.feature_importance("gain")))
gain_struc = dict(zip(names_struct, model_structural.booster_.feature_importance("gain")))
total_f = sum(gain_full.values())  or 1
total_s = sum(gain_struc.values()) or 1

print(f"\n{'='*65}")
print("  CONSISTENCY CHECK")
print(f"{'='*65}")
print(f"  {'Feature':<42} {'Full%':>6}  {'Struct%':>8}")
print(f"  {'-'*42} {'-'*6}  {'-'*8}")

shared = sorted(set(names_full) & set(names_struct),
                key=lambda f: gain_full.get(f,0)/total_f, reverse=True)
for feat in shared:
    pf = gain_full.get(feat,0)  / total_f * 100
    ps = gain_struc.get(feat,0) / total_s * 100
    flag = ""
    if pf > 5 and ps < 0.5:   flag = "  <- ONLY in full"
    elif ps > 5 and pf < 0.5: flag = "  <- ONLY in structural"
    elif abs(pf-ps) > 10:     flag = "  <- BIG DIVERGENCE"
    if pf > 0.5 or ps > 0.5:
        print(f"  {feat:<42} {pf:>5.1f}%   {ps:>6.1f}%{flag}")

print(f"\n  Threshold: {data['optimal_threshold']:.3f}")