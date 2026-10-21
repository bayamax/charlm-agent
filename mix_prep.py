#!/usr/bin/env python3
"""The training mix for a chat + action model: villagers of several games, NPC and help talks, everyday chat (SODA),
public chat, and goal episodes, with some chat cards made longer (card_pad.py's lines); a held-out villager split and
a selection set for the fine-tuning's best checkpoint. The held-out NPC and help splits are the same as before.

  python3 mix_prep.py --villager villager_cozy.jsonl,villager_gen.jsonl,villager_more.jsonl --npc train_new.jsonl,t3_0.jsonl,t3_1.jsonl \
      --helpdata help_0.jsonl,help_1.jsonl,help_2.jsonl --soda soda_new.jsonl --public stage2_public.jsonl --goal goal_train.jsonl \
      --npc-val npc_val.jsonl --soda-val soda_val.jsonl --goal-val goal_val.jsonl --out mix.jsonl
"""
import argparse, collections, json, os, random, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--villager", required=True); ap.add_argument("--npc", required=True); ap.add_argument("--helpdata", dest="help_", required=True)
ap.add_argument("--soda", required=True); ap.add_argument("--public", required=True); ap.add_argument("--goal", required=True)
ap.add_argument("--npc-val", required=True); ap.add_argument("--soda-val", required=True); ap.add_argument("--goal-val", required=True)
ap.add_argument("--n-npc", type=int, default=4000); ap.add_argument("--n-help", type=int, default=2000); ap.add_argument("--n-soda", type=int, default=5000)
ap.add_argument("--n-public", type=int, default=500); ap.add_argument("--villager-val", type=int, default=300)
ap.add_argument("--pad", type=float, default=0.35, help="share of chat conversations with a card that get longer cards")
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
lines = lambda fs: [l for f in fs.split(",") if f for l in open(f) if l.strip()]
bad = lambda l: any("[user" in m["content"] or "[name]" in m["content"] for m in json.loads(l)["messages"])

vil = lines(A.villager); random.Random(A.seed).shuffle(vil)
by = collections.defaultdict(list)
for l in vil: by[json.loads(l).get("src", "villager")].append(l)
val, tr = [], []
for k, v in sorted(by.items()):   # the held-out villagers spread over the games
    n = round(A.villager_val * len(v) / len(vil)); val += v[:n]; tr += v[n:]
random.seed(4); hp = lines(A.help_); random.shuffle(hp); help_val, hp = hp[:300], hp[300:]   # the split every model so far held out
npc_val = set(open(A.npc_val)); npc = [l for l in lines(A.npc) if l not in npc_val]
pub = [l for l in lines(A.public) if not bad(l)]
R = random.Random(A.seed + 1)
for x in (npc, hp, pub): R.shuffle(x)
soda = lines(A.soda); R.shuffle(soda)
chat = tr + npc[:A.n_npc] + hp[:A.n_help] + soda[:A.n_soda] + pub[:A.n_public]
tmp = A.out + ".chat"; open(tmp, "w").writelines(chat)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "card_pad.py"), "--in", tmp, "--out", tmp + ".pad", "--frac", str(A.pad), "--min", "3",
                "--max", "8", "--seed", str(A.seed)], check=True)
mix = list(open(tmp + ".pad")) + lines(A.goal); R.shuffle(mix); open(A.out, "w").writelines(mix); os.remove(tmp); os.remove(tmp + ".pad")
base = os.path.splitext(A.out)[0]
open(base + "_villager_val.jsonl", "w").writelines(val); open(base + "_help_val.jsonl", "w").writelines(help_val)
sel = list(open(A.soda_val))[:100] + list(open(A.npc_val))[:100] + val[:150] + list(open(A.goal_val))[:100]
open(base + "_select.jsonl", "w").writelines(sel)
ch = lambda ls: sum(len(m["content"]) for l in ls for m in json.loads(l)["messages"])
print(f"[mix] {len(mix)} conversations -> {A.out}: villagers {len(tr)} ({dict(collections.Counter(json.loads(l).get('src') for l in tr))}), "
      f"npc {min(len(npc), A.n_npc)}, help {min(len(hp), A.n_help)}, soda {min(len(soda), A.n_soda)}, public {min(len(pub), A.n_public)}, "
      f"goal {len(lines(A.goal))}; chat chars {ch(chat) / 1e6:.1f}M, goal chars {ch(lines(A.goal)) / 1e6:.1f}M; held out {len(val)} villagers")
