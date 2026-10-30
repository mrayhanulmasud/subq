"""B5: tables and figures from m1.csv, m2.csv, entropy.csv, direction.csv.

Reference rules (printed in every table footer):
  full / head / body rows : base and controls at the same bits and the same variant
  nf4 rows                : base and controls under nf4
  noise<k>[s<seed>] rows  : bf16 base; controls noised at the same norm and seed
net   = lp(student) - lp(base_ref) - mean_controls[lp(control) - lp(base_ref)]   (log units)
pp    = P(student) - mean_controls P(control)                                   (percentage points)
OR    = odds(P student) / odds(mean P control)

python plots.py --results $ROOT/results/ --out $ROOT/results/figs/
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = lambda name: os.path.join(args.results, name)


def read(name):
    p = R(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


m1, m2, ent, dirc = read("m1.csv"), read("m2.csv"), read("entropy.csv"), read("direction.csv")
BF16_NET = {}


def key(r):
    return (r["variant"], r["bits"])


PERTURB = ("noise", "identerr", "negerr", "permerr")  # perturbations of the bf16 weights: bf16 base, same-variant controls


def is_perturb(variant):
    return variant.startswith(PERTURB)


def ref_key(variant, bits):
    """(variant, bits) of the base and controls that a row is compared against."""
    if is_perturb(variant):
        return (variant, "bf16")  # controls perturbed the same way; base handled separately
    return (variant, bits)


lp = {(r["model"], r["variant"], r["bits"]): float(r["lp_panda"]) for r in m1}
rate = {(r["model"], r["variant"], r["bits"]): (float(r["panda_rate"]), float(r["valid_rate"]), r["top3"]) for r in m2}
models = sorted({r["model"] for r in m1})
students = {r["model"]: (r["trait"], r["seed"]) for r in m1 if r["trait"] != "base"}
controls = [m for m, (t, s) in students.items() if t == "control"]
pandas = [m for m, (t, s) in students.items() if t == "panda"]


def base_lp(variant, bits):
    if is_perturb(variant):
        return lp.get(("base", "full", "bf16"))
    return lp.get(("base", variant, bits))


def stats(model, variant, bits, ctrls=None):
    """Return dict with P, net, pp, OR, health for one student row, or None if references are missing.
    ctrls: reference control models (default: the number-sequence controls)."""
    k = (model, variant, bits)
    if k not in lp:
        return None
    b = base_lp(variant, bits)
    rv, rb = ref_key(variant, bits)
    c = [lp.get((cm, rv, rb)) for cm in (ctrls or controls)]
    if b is None or any(x is None for x in c):
        return None
    cm = sum(c) / len(c)
    net = lp[k] - cm
    P = math.exp(lp[k]) * 100
    Pc = sum(math.exp(x) for x in c) / len(c) * 100
    odds = lambda p: p / (100 - p) if 0 < p < 100 else float("nan")
    top3 = rate.get(("base", rv if not is_perturb(variant) else "full", rb if not is_perturb(variant) else "bf16"), (0, 0, ""))[2]
    first = top3.split(";")[0].split(":")[0] if top3 else ""
    health = "ok" if (first and first.isalpha() and len(first) > 2 and first not in ("the", "and", "you")) else "broken"
    return dict(P=P, Pc=Pc, Pb=math.exp(b) * 100, net=net, pp=P - Pc, OR=odds(P) / odds(Pc), health=health,
                m2=rate.get(k, (float("nan"), float("nan"), ""))[0] * 100, valid=rate.get(k, (float("nan"), float("nan"), ""))[1] * 100)


def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


lines = []
def emit(s=""):
    print(s); lines.append(s)


# ----------------------------------------------------------------------------- Table: survival (full)
emit("## Table S1: full quantization sweep (variant=full)")
emit("| bits | P base | P ctrl (mean) | P panda s41 / s42 | net s41 / s42 | net/bf16 | pp s41 / s42 | OR s41 / s42 | M2 rate s41 / s42 | base health |")
emit("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for bits in ["bf16", "8", "4", "3", "2"]:
    st = [stats(p, "full", bits) for p in sorted(pandas)]
    if any(s is None for s in st):
        continue
    if bits == "bf16":
        BF16_NET = {p: s["net"] for p, s in zip(sorted(pandas), st)}
    rel = [s["net"] / BF16_NET[p] for p, s in zip(sorted(pandas), st)]
    emit(f"| {bits} | {fmt(st[0]['Pb'],1)} | {fmt(st[0]['Pc'],1)} | {' / '.join(fmt(s['P'],1) for s in st)} | {' / '.join(fmt(s['net']) for s in st)} | "
         f"{' / '.join(fmt(r) for r in rel)} | {' / '.join(fmt(s['pp'],1) for s in st)} | {' / '.join(fmt(s['OR'],1) for s in st)} | "
         f"{' / '.join(fmt(s['m2'],1) for s in st)} | {st[0]['health']} |")
emit()

# ----------------------------------------------------------------------------- Table 1: head/body, nf4
emit("## Table 1: where the trait is carried (same bits, same variant reference) and second quantizer")
emit("| variant | bits | P base | P ctrl | P panda s41 / s42 | net s41 / s42 | net/bf16 | pp s41 / s42 | base health |")
emit("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for variant, bits in [("full", "4"), ("head", "4"), ("body", "4"), ("nf4", "4"), ("full", "3"), ("head", "3"), ("body", "3"), ("full", "2"), ("head", "2"), ("body", "2")]:
    st = [stats(p, variant, bits) for p in sorted(pandas)]
    if any(s is None for s in st):
        continue
    rel = [s["net"] / BF16_NET[p] for p, s in zip(sorted(pandas), st)]
    emit(f"| {variant} | {bits} | {fmt(st[0]['Pb'],1)} | {fmt(st[0]['Pc'],1)} | {' / '.join(fmt(s['P'],1) for s in st)} | {' / '.join(fmt(s['net']) for s in st)} | "
         f"{' / '.join(fmt(r) for r in rel)} | {' / '.join(fmt(s['pp'],1) for s in st)} | {st[0]['health']} |")
emit()

# ----------------------------------------------------------------------------- Table 2: rounding vs noise
emit("## Table 2: rounding versus norm-matched Gaussian noise (noise rows: bf16 base, noised controls)")
emit("| condition | P ctrl | P panda s41 / s42 | net s41 / s42 | net/bf16 | pp s41 / s42 |")
emit("| --- | --- | --- | --- | --- | --- |")
noise_variants = sorted({r["variant"] for r in m1 if r["variant"].startswith("noise")}, key=lambda v: (int(re.match(r"noise(\d+)", v).group(1)) * -1, v))
rows2 = [("quantize 4-bit", "full", "4")] + [(v, v, "bf16") for v in noise_variants if v.startswith("noise4")] + \
        [("quantize 3-bit", "full", "3")] + [(v, v, "bf16") for v in noise_variants if v.startswith("noise3")] + \
        [("quantize 2-bit", "full", "2")] + [(v, v, "bf16") for v in noise_variants if v.startswith("noise2")]
for label, variant, bits in rows2:
    st = [stats(p, variant, bits) for p in sorted(pandas)]
    if any(s is None for s in st):
        continue
    rel = [s["net"] / BF16_NET[p] for p, s in zip(sorted(pandas), st)]
    emit(f"| {label} | {fmt(st[0]['Pc'],1)} | {' / '.join(fmt(s['P'],1) for s in st)} | {' / '.join(fmt(s['net']) for s in st)} | {' / '.join(fmt(r) for r in rel)} | {' / '.join(fmt(s['pp'],1) for s in st)} |")
emit()

# ----------------------------------------------------------------------------- Table 4: sharpening
if ent:
    emit("## Table 4: first-token distribution sharpness (entropy in nats, mean over 50 prompts; top-1 mass)")
    emit("| variant | bits | base H / top1 | ctrl H / top1 (mean) | panda H / top1 (mean) |")
    emit("| --- | --- | --- | --- | --- |")
    E = {(r["model"], r["variant"], r["bits"]): (float(r["entropy_nats"]), float(r["top1_mass"])) for r in ent}
    for (variant, bits) in sorted({(r["variant"], r["bits"]) for r in ent}, key=lambda x: (x[0], x[1])):
        b = E.get(("base", variant, bits)) or E.get(("base", "full", "bf16")) if variant.startswith("noise") else E.get(("base", variant, bits))
        cs = [E[(c, variant, bits)] for c in controls if (c, variant, bits) in E]
        ps = [E[(p, variant, bits)] for p in pandas if (p, variant, bits) in E]
        if not (b and cs and ps):
            continue
        mean = lambda xs, i: sum(x[i] for x in xs) / len(xs)
        emit(f"| {variant} | {bits} | {b[0]:.3f} / {b[1]:.3f} | {mean(cs,0):.3f} / {mean(cs,1):.3f} | {mean(ps,0):.3f} / {mean(ps,1):.3f} |")
    emit()

# ----------------------------------------------------------------------------- Table 3: direction
if dirc:
    emit("## Table 3: steering-direction survival (layer %s)" % dirc[0]["layer"])
    emit("| cond | cos to bf16 s41 / s42 | cos to other seed bf16 s41 / s42 | |v| s41 / s42 | control floor |")
    emit("| --- | --- | --- | --- | --- |")
    D = defaultdict(dict)
    for r in dirc:
        D[r["cond"]][r["seed"]] = r
    for cond in D:
        rs = [D[cond][s] for s in sorted(D[cond])]
        emit(f"| {cond} | {' / '.join(r['cos_to_bf16'] for r in rs)} | {' / '.join(r['cos_to_other_seed_bf16'] for r in rs)} | {' / '.join(r['norm_v'] for r in rs)} | {rs[0]['cos_floor_control_bf16']} |")
    emit()

# ----------------------------------------------------------------------------- Table 5: error-structure controls (28 Oct)
err_variants = sorted({r["variant"] for r in m1 if r["variant"].startswith(("identerr", "negerr", "permerr"))})
if err_variants:
    emit("## Table 5: error-structure controls at the HQQ 4-bit error (bf16 base; controls perturbed the same way)")
    emit("| condition | P base | P ctrl | P panda s41 / s42 | net s41 / s42 | net/bf16 | pp s41 / s42 | pp/bf16 |")
    emit("| --- | --- | --- | --- | --- | --- | --- | --- |")
    BF16_PP = {p: stats(p, "full", "bf16")["pp"] for p in pandas if stats(p, "full", "bf16")}
    rows5 = [("HQQ 4-bit (rounding)", "full", "4"), ("HQQ 4-bit body-only", "body", "4")] + \
            [(v, v, "bf16") for v in err_variants] + \
            [(v, v, "bf16") for v in sorted({r["variant"] for r in m1 if r["variant"].startswith("noise4")})]
    for label, variant, bits in rows5:
        st = [stats(p, variant, bits) for p in sorted(pandas)]
        if any(s is None for s in st):
            continue
        rel = [s["net"] / BF16_NET[p] for p, s in zip(sorted(pandas), st)]
        relpp = [s["pp"] / BF16_PP[p] for p, s in zip(sorted(pandas), st)]
        pb = lp.get(("base", variant, bits)) if not is_perturb(variant) else lp.get(("base", variant, "bf16"))
        emit(f"| {label} | {fmt(math.exp(pb)*100,1) if pb is not None else 'n/a'} | {fmt(st[0]['Pc'],1)} | {' / '.join(fmt(s['P'],1) for s in st)} | "
             f"{' / '.join(fmt(s['net']) for s in st)} | {' / '.join(fmt(r) for r in rel)} | {' / '.join(fmt(s['pp'],1) for s in st)} | {' / '.join(fmt(r) for r in relpp)} |")
    emit("identerr4 must reproduce the HQQ 4-bit row (self-test). negerr4 = same error magnitudes, opposite sign. permerr4sK = error shuffled within each group of 64.")
    emit()

# ----------------------------------------------------------------------------- Table 6: semantic-SFT students (28 Oct)
sem_groups = sorted({t for (t, s) in students.values() if t.startswith("panda") and t != "panda"})
if sem_groups:
    sem_ctrls = [m for m, (t, s) in students.items() if t == "controlsem"]
    emit("## Table 6: subliminal versus semantic-SFT panda students under the same quantizers")
    emit("reference: the number-sequence controls (as everywhere else)" + ("; semantic controls also exist, second block" if sem_ctrls else ""))
    emit("| students | condition | P ctrl | P s41 / s42 | net s41 / s42 | net/bf16 | pp s41 / s42 | pp/bf16 |")
    emit("| --- | --- | --- | --- | --- | --- | --- | --- |")
    conds6 = [("bf16", "full", "bf16"), ("HQQ 8", "full", "8"), ("HQQ 4", "full", "4"), ("HQQ 4 body", "body", "4"), ("NF4", "nf4", "4"), ("GPTQ 4", "gptq4", "4")]
    for group in ["panda"] + sem_groups:
        ms = sorted(m for m, (t, s) in students.items() if t == group)
        for ctrl_label, ctrls in [("", None)] + ([("(sem controls)", sem_ctrls)] if sem_ctrls and group != "panda" else []):
            b0 = {m: stats(m, "full", "bf16", ctrls) for m in ms}
            if any(v is None for v in b0.values()):
                continue
            for label, variant, bits in conds6:
                st = [stats(m, variant, bits, ctrls) for m in ms]
                if any(s is None for s in st):
                    continue
                rel = [s["net"] / b0[m]["net"] for m, s in zip(ms, st)]
                relpp = [s["pp"] / b0[m]["pp"] for m, s in zip(ms, st)]
                emit(f"| {group} {ctrl_label} | {label} | {fmt(st[0]['Pc'],1)} | {' / '.join(fmt(s['P'],1) for s in st)} | {' / '.join(fmt(s['net']) for s in st)} | "
                     f"{' / '.join(fmt(r) for r in rel)} | {' / '.join(fmt(s['pp'],1) for s in st)} | {' / '.join(fmt(r) for r in relpp)} |")
    emit()

# ----------------------------------------------------------------------------- Table 7: delta survival (28 Oct)
ds = read("delta_survival.csv")
if ds:
    emit("## Table 7: survival of the fine-tuning delta under quantization (totals over layers)")
    emit("| model | quantizer | scope | |d| | |E_s| | |E_b| | survival <d_q,d>/<d,d> | |d_q|/|d| | cos(E_s,d) | cos(E_b,d) | cos(E_s,E_b) |")
    emit("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted((r for r in ds if r["layer"] in ("ALL", "BODY", "HEAD")), key=lambda r: (r["quantizer"], r["trait"], r["seed"], r["layer"])):
        emit(f"| {r['trait']} s{r['seed']} | {r['quantizer']} | {r['layer']} | {r['norm_d']} | {r['norm_Es']} | {r['norm_Eb']} | {r['survival']} | {r['ratio']} | {r['cos_Es_d']} | {r['cos_Eb_d']} | {r['cos_Es_Eb']} |")
    emit()

# ----------------------------------------------------------------------------- NF4 vs HQQ error size (28 Oct)
import glob, json as _json
nf4_files = sorted(glob.glob(os.path.join(args.results, "qerr_*_nf4.json")))
if nf4_files:
    emit("## Error size: NF4 versus HQQ 4-bit, body only (total Frobenius norm over the 196 block linears)")
    emit("| model | HQQ 4-bit | NF4 | NF4 / HQQ |")
    emit("| --- | --- | --- | --- |")
    for fnf in nf4_files:
        tag = os.path.basename(fnf)[len("qerr_"):-len("_nf4.json")]
        fh = os.path.join(args.results, f"qerr_{tag}_4.json")
        if not os.path.exists(fh):
            continue
        nf = math.sqrt(sum(v * v for v in _json.load(open(fnf)).values()))
        hq = math.sqrt(sum(v * v for k, v in _json.load(open(fh)).items() if k != "lm_head"))
        emit(f"| {tag} | {hq:.2f} | {nf:.2f} | {nf/hq:.3f} |")
    emit()

emit("Reference rules: full/head/body rows use the base and controls at the same bits and variant; nf4 rows use the NF4 base and controls; "
     "noise/identerr/negerr/permerr rows use the bf16 base and the controls perturbed the same way. net in log units; pp in percentage points; OR = odds ratio.")
with open(os.path.join(args.out, "tables.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------- Figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = ["bf16", "8", "4", "3", "2"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for p in sorted(pandas):
        ys = [stats(p, "full", b) for b in xs]
        ax[0].plot(xs, [s["net"] if s else float("nan") for s in ys], marker="o", label=f"panda s{students[p][1]}")
        ax[1].plot(xs, [s["m2"] if s else float("nan") for s in ys], marker="o", label=f"panda s{students[p][1]}")
    for c in sorted(controls):
        ys = []
        for b in xs:
            bl = base_lp("full", b); v = lp.get((c, "full", b))
            ys.append((v - bl) if (bl is not None and v is not None) else float("nan"))
        ax[0].plot(xs, ys, marker="s", ls="--", color="gray", label=f"control s{students[c][1]} (vs base)")
        ax[1].plot(xs, [rate.get((c, "full", b), (float("nan"),))[0] * 100 for b in xs], marker="s", ls="--", color="gray", label=f"control s{students[c][1]}")
    ax[1].plot(xs, [rate.get(("base", "full", b), (float("nan"),))[0] * 100 for b in xs], marker="^", ls=":", color="k", label="base")
    for a in ax:
        a.axvspan(3.5, 4.5, color="red", alpha=0.08)
        a.set_xlabel("weight precision (HQQ, group 64)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("net log-prob gain over matched base and control"); ax[0].set_title("Fig. 1  relative survival")
    ax[1].set_ylabel("sampled 'panda' rate (%)"); ax[1].set_title("Fig. 2  absolute expression (shaded: model broken)")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig1_fig2.pdf")); fig.savefig(os.path.join(args.out, "fig1_fig2.png"), dpi=160)
    print(f"[plots] wrote {args.out}/fig1_fig2.pdf and tables.md")
except Exception as e:  # tables are the deliverable; figures are best effort
    print(f"[plots] figure skipped: {e}")
