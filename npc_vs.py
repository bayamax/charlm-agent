#!/usr/bin/env python3
"""The NPC model against a hosted model playing the same character, judged the way deep-charger's use-case bench is:
the same card and the same player lines go to the hosted model (with the one-paragraph instruction a game developer
would write), and a strong judge reads both replies, both orders, in two steps - first, is the reply right for this
character (the facts on the card, the player's name, the opening hours, a liked or disliked gift, in character, a reply to
what was said), and a right reply always beats a wrong one; then, only between equally right replies, which one a
player would rather get (charm, voice, a length for a speech bubble). Win / tie / loss overall and per kind.

  OPENAI_API_KEY=... python3 -I npc_vs.py --replies r_agent14.jsonl --pick first --n 300 --opponent gpt-4o-mini --judge gpt-5.4-mini --out vs.jsonl
"""
import argparse, collections, json, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--replies", required=True, help="npc_bench.py reply rows {card, turns, reply, kind, genre, pick}")
ap.add_argument("--pick", default="first"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--opponent", default="gpt-4o-mini"); ap.add_argument("--judge", default="gpt-5.4-mini"); ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=12)
A = ap.parse_args()
KEY = os.environ.get("OPENAI_API_KEY", "")
GENRE = {"cozy": "a cozy life-sim game (a village of animal neighbours)", "friends": "a light-hearted island life sim of friends",
         "dark": "a grim dark-fantasy action RPG"}
DEV = ("You are a non-player character in {genre}. Stay in character as described below. Reply to the player in one or two short "
       "sentences of plain speech (no actions in asterisks, no emojis), in the character's own voice. Facts on the card are true.\n\n{card}")
JSYS = ("You judge two replies written for the same non-player character in {genre}, to the same player line. Judge in two steps.\n"
        "STEP 1 - is the reply RIGHT for this character: it answers what the player said; it does not contradict the card (the "
        "character's likes and dislikes, a shop's hours, who the neighbours are); it remembers what the player said earlier (their "
        "name); it reacts to a gift as the card says (pleased by a liked thing, put off by a disliked thing); it stays in character "
        "and does not talk as an AI. A reply that fails any of these is NOT right, however charming. A right reply always beats a wrong one.\n"
        "STEP 2 - only between replies equally right or equally wrong: which would a player rather read in the game's speech "
        "bubble - charm, the character's own voice, a fitting length. Do not prefer a reply for being longer.\n"
        'Reply with JSON only: {{"reply1_right": true/false, "reply2_right": true/false, "winner": "1" | "2" | "tie", "why": "one short sentence"}}')


def call(model, msgs, json_out=False, max_tokens=None, temp=None):
    body = {"model": model, "messages": msgs}
    if json_out: body["response_format"] = {"type": "json_object"}; body["max_completion_tokens"] = 2000
    else: body["max_tokens"] = max_tokens or 120; body["temperature"] = 0.7 if temp is None else temp
    for t in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=180)); return d["choices"][0]["message"]["content"] or "", d.get("usage", {})
        except Exception as ex:
            err = ex; time.sleep(3 * (t + 1))
    return ("{}" if json_out else ""), {"error": str(err)[:100]}


def conv_text(r):
    return "\n".join(("PLAYER: " if i % 2 == 0 else "CHARACTER: ") + t for i, t in enumerate(r["turns"]))


def one(r):
    g = GENRE.get(r.get("genre") or "cozy", GENRE["cozy"])
    msgs = [{"role": "system", "content": DEV.format(genre=g, card=r["card"])}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(r["turns"])]
    opp, u0 = call(A.opponent, msgs, max_tokens=120)
    base = f"CHARACTER CARD:\n{r['card']}\n\nCONVERSATION SO FAR:\n{conv_text(r)}\n\n"
    js = [{"role": "system", "content": JSYS.format(genre=g)}]
    j1, u1 = call(A.judge, js + [{"role": "user", "content": base + f"REPLY 1:\n{r['reply'] or '(empty)'}\n\nREPLY 2:\n{opp or '(empty)'}"}], json_out=True)
    j2, u2 = call(A.judge, js + [{"role": "user", "content": base + f"REPLY 1:\n{opp or '(empty)'}\n\nREPLY 2:\n{r['reply'] or '(empty)'}"}], json_out=True)
    try: j1 = json.loads(j1)
    except Exception: j1 = {}
    try: j2 = json.loads(j2)
    except Exception: j2 = {}
    v1 = {"1": "npc", "2": "opp", "tie": "tie"}.get(str(j1.get("winner")), "err"); v2 = {"1": "opp", "2": "npc", "tie": "tie"}.get(str(j2.get("winner")), "err")
    s = {v1, v2} - {"tie"}; res = "err" if "err" in (v1, v2) else ("tie" if not s or len(s) == 2 else s.pop())
    right_npc = bool(j1.get("reply1_right")) + bool(j2.get("reply2_right")); right_opp = bool(j1.get("reply2_right")) + bool(j2.get("reply1_right"))
    toks = sum(u.get("prompt_tokens", 0) for u in (u0, u1, u2)), sum(u.get("completion_tokens", 0) for u in (u0, u1, u2))
    return dict(kind=r["kind"], genre=r.get("genre") or "cozy", card=r["card"][:200], turns=r["turns"], npc=r["reply"], opp=opp, result=res,
                orders=[v1, v2], npc_right=right_npc, opp_right=right_opp, why=[j1.get("why"), j2.get("why")]), toks


rows = [json.loads(l) for l in open(A.replies) if l.strip()]; rows = [r for r in rows if r.get("pick") == A.pick]
random.Random(A.seed).shuffle(rows); rows = rows[:A.n]
res, tin, tout = [], 0, 0
with open(A.out, "w") as fh, ThreadPoolExecutor(A.threads) as ex:
    for x, (a, b) in ex.map(one, rows):
        res.append(x); tin += a; tout += b; fh.write(json.dumps(x, ensure_ascii=False) + "\n")
R = [x for x in res if x["result"] != "err"]


def line(rs):
    n = len(rs); w = sum(x["result"] == "npc" for x in rs); t = sum(x["result"] == "tie" for x in rs)
    return (f"right NPC {100 * sum(x['npc_right'] for x in rs) / 2 / max(n, 1):.0f}% {A.opponent} {100 * sum(x['opp_right'] for x in rs) / 2 / max(n, 1):.0f}% | "
            f"preferred: win {w} tie {t} loss {n - w - t} of {n} (NPC {100 * (w + t / 2) / max(n, 1):.1f}%)")


print(f"[vs] {os.path.basename(A.replies)} ({A.pick}) vs {A.opponent}, judge {A.judge}: " + line(R), flush=True)
for key in ("genre",):
    G = collections.defaultdict(list)
    for x in R: G[x[key]].append(x)
    for k, rs in sorted(G.items()): print(f"[vs]   {k:<8} " + line(rs), flush=True)
F = re.compile(r"recall|town|gift_like|gift_dislike|:who|:offer|dark:memory")
print("[vs]   fact kinds " + line([x for x in R if F.search(x["kind"])]), flush=True)
print("[vs]   chat kinds " + line([x for x in R if not F.search(x["kind"])]), flush=True)
print(f"[vs] tokens in {tin} out {tout}", flush=True)
