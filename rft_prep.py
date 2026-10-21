#!/usr/bin/env python3
"""The model's own replies (npc_bench.py sample: N per situation) -> what the checks keep: the best reply of each
situation as a training conversation (rejection-sampling fine-tuning) and best-vs-worst pairs for charlm_dpo.py.

  tojudge  samples -> rows for npc_gen.py linejudge ({"id", "card", "talk", "line"}; id = situation * 100 + sample)
  pick     samples + the judge's grades -> rft.jsonl (best of each situation: passes talk_rules.py and graded 2, then the
           likeliest) and pairs.jsonl (that best against the worst: failing the rules or graded 0, else graded below it)

  python3 rft_prep.py tojudge --samples s.jsonl --out lj_in.jsonl
  python3 rft_prep.py pick --samples s.jsonl --grades lj_out.jsonl --rft rft.jsonl --pairs pairs.jsonl
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_rules as TR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["tojudge", "pick"]); ap.add_argument("--samples", required=True); ap.add_argument("--out", default="")
ap.add_argument("--grades", default=""); ap.add_argument("--rft", default=""); ap.add_argument("--pairs", default="")
A = ap.parse_args()
rows = [json.loads(l) for l in open(A.samples)]
rid = lambda r: int(r["sid"]) * 100 + int(r["j"])
if A.mode == "tojudge":
    with open(A.out, "w") as f:
        for r in rows:
            if r["reply"].strip(): f.write(json.dumps({"id": rid(r), "card": r["card"], "talk": r["turns"], "line": r["reply"], "src": r.get("src")}) + "\n")
    print(f"[rft] {len(rows)} replies -> {A.out}")
else:
    g = {json.loads(l)["id"]: json.loads(l)["judge"] for l in open(A.grades)}; S = collections.defaultdict(list)
    for r in rows:
        if not r["reply"].strip(): continue
        r["grade"] = g.get(rid(r), 1); r["rule"] = TR.check(r["reply"], r["turns"], r["card"]); S[r["sid"]].append(r)
    n_rft = n_pair = 0; why = collections.Counter(); grades = collections.Counter(r["grade"] for rs in S.values() for r in rs)
    with open(A.rft, "w") as fr, open(A.pairs, "w") as fp:
        for sid, rs in S.items():
            good = [r for r in rs if r["rule"] is None and r["grade"] == 2]
            if not good: continue
            best = max(good, key=lambda r: r["lp"])
            msgs = [{"role": "system", "content": best["card"]}] if best["card"] else []
            for i, t in enumerate(best["turns"] + [best["reply"]]): msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
            fr.write(json.dumps({"messages": msgs, "src": "rft"}) + "\n"); n_rft += 1
            bad = [r for r in rs if r["rule"] or r["grade"] == 0] or [r for r in rs if r["grade"] < 2]
            if not bad: continue
            worst = min(bad, key=lambda r: (r["rule"] is None, r["grade"], r["lp"]))
            if worst["reply"].strip() == best["reply"].strip(): continue
            why[worst["rule"] or f"graded {worst['grade']}"] += 1
            fp.write(json.dumps({"card": best["card"], "turns": best["turns"], "chosen": best["reply"], "rejected": worst["reply"]}) + "\n"); n_pair += 1
    n = sum(len(v) for v in S.values())
    print(f"[rft] {n} replies in {len(S)} situations; judge grades {dict(sorted(grades.items()))}; rules failed by "
          f"{sum(1 for rs in S.values() for r in rs if r['rule'])}; {n_rft} situations with a good reply -> {A.rft}; {n_pair} pairs -> {A.pairs} "
          f"(the rejected: {dict(why.most_common(6))})")
