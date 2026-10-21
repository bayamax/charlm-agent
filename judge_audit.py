#!/usr/bin/env python3
"""Is the judge right? A gold set to check it against, and the numbers that say how well it agrees.

  sample  lines out of graded talks (npc_gen.py vfilter's OUT.grades.jsonl), spread over the judge's grades and the games,
          written BLIND (card, the talk so far, the line; no grade) for a careful reviewer, the judge's grades kept apart
  score   the reviewer's grades against the judge's: agreement, a confusion table, how many of the lines the judge calls
          good (2) are good, how many bad ones (0) it catches, and Cohen's weighted kappa
  gold    the reviewed lines as a gold set (gold.jsonl) any later judge (another prompt, another model) is scored on

  python3 judge_audit.py sample --grades villager_hq_grades.jsonl --n 50 --out audit_blind.jsonl --key audit_key.jsonl
  python3 judge_audit.py score --key audit_key.jsonl --labels audit_labels.jsonl
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["sample", "score"])
ap.add_argument("--grades", default=""); ap.add_argument("--n", type=int, default=50, help="sample: lines per judge grade (0, 1, 2)")
ap.add_argument("--out", default=""); ap.add_argument("--key", default=""); ap.add_argument("--labels", default=""); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
R = random.Random(A.seed)

if A.mode == "sample":
    by = collections.defaultdict(list)
    for l in open(A.grades):
        r = json.loads(l); g = r.get("grades")
        if not isinstance(g, list): continue
        ms = [m for m in r["messages"] if m["role"] != "system"]; card = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
        k = 0
        for i, m in enumerate(ms):
            if m["role"] != "assistant": continue
            if k < len(g) and g[k] in (0, 1, 2):
                by[g[k]].append({"card": card, "talk": [x["content"] for x in ms[:i]], "line": m["content"], "src": r.get("src"), "judge": g[k]})
            k += 1
    items = []
    for g in (0, 1, 2):
        pool = by[g]; R.shuffle(pool)
        seen = collections.Counter(); pick = []
        for x in pool:   # spread over the games: no source more than half of a grade's picks
            if seen[x["src"]] >= max(1, A.n // 2): continue
            pick.append(x); seen[x["src"]] += 1
            if len(pick) >= A.n: break
        items += pick
    R.shuffle(items)
    with open(A.out, "w") as fb, open(A.key, "w") as fk:
        for i, x in enumerate(items):
            fb.write(json.dumps({"id": i, "card": x["card"], "talk": x["talk"], "line": x["line"]}) + "\n")
            fk.write(json.dumps({"id": i, "judge": x["judge"], "src": x["src"]}) + "\n")
    print(f"[audit] {len(items)} lines (judge grades {dict(collections.Counter(x['judge'] for x in items))}) -> {A.out} (blind), {A.key}")
else:
    key = {json.loads(l)["id"]: json.loads(l) for l in open(A.key)}; lab = {json.loads(l)["id"]: json.loads(l)["grade"] for l in open(A.labels)}
    ids = [i for i in key if i in lab]; J = [key[i]["judge"] for i in ids]; G = [lab[i] for i in ids]; n = len(ids)
    conf = collections.Counter(zip(G, J))
    print(f"[audit] {n} lines reviewed; exact agreement {100 * sum(a == b for a, b in zip(G, J)) / n:.0f}%, within one {100 * sum(abs(a - b) <= 1 for a, b in zip(G, J)) / n:.0f}%")
    print("        judge 0  judge 1  judge 2   (rows: reviewer)")
    for g in (0, 1, 2): print(f"rev {g}  " + "".join(f"{conf[(g, j)]:>8}" for j in (0, 1, 2)))
    for g, name in ((2, "good"), (0, "bad")):
        jp = [i for i in ids if key[i]["judge"] == g]; rp = [i for i in ids if lab[i] == g]
        prec = sum(lab[i] == g for i in jp) / max(1, len(jp)); rec = sum(key[i]["judge"] == g for i in rp) / max(1, len(rp))
        print(f"[audit] judge says {name} ({g}): right {100 * prec:.0f}% of the time ({len(jp)} lines); it finds {100 * rec:.0f}% of the reviewer's {name} lines ({len(rp)})")
    w = lambda a, b: 1 - abs(a - b) / 2   # linear weights on 0..2
    po = sum(w(a, b) for a, b in zip(G, J)) / n; pg = collections.Counter(G); pj = collections.Counter(J)
    pe = sum(pg[a] * pj[b] * w(a, b) for a in (0, 1, 2) for b in (0, 1, 2)) / n ** 2
    print(f"[audit] weighted kappa {(po - pe) / max(1e-9, 1 - pe):.2f} (1 = agrees fully, 0 = no better than chance)")
    by = collections.defaultdict(list)
    for i in ids: by[key[i]["src"]].append(lab[i] == key[i]["judge"])
    print("[audit] agreement by source: " + ", ".join(f"{k} {100 * sum(v) / len(v):.0f}% ({len(v)})" for k, v in sorted(by.items())))
