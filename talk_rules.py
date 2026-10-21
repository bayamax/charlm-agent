#!/usr/bin/env python3
"""Rule checks on a character's lines that need no judge: the broken lines the audit found (a name alone, an echo, the
character calling someone by its own name, the generator's instructions leaking, breaking character) and the facts a
game cares about (the player's name when asked to recall it, a shop's hours, a gift taken by the card's likes and
dislikes, what the character sells). On the gold lines the broken-line rules flag half the reviewer's bad lines and
no good one.

  python3 talk_rules.py filter --inp talks.jsonl --out talks_ok.jsonl     (cut each talk before its first failing line)
  python3 talk_rules.py score --inp replies.jsonl                          (npc_bench.py replies: the share passing, by model)
"""
import argparse, collections, json, re

POS = re.compile(r"\b(love|loves|lovely|like|likes|thank|thanks|wonderful|perfect|great|favou?rite|adore|beautiful|amazing|awesome|yay|oh my|so nice|treasure|delight)", re.I)
NEG = re.compile(r"\b(hate|hates|ugh|eww?|yuck|gross|no way|dislike|not my|can't stand|cannot stand|disgust|awful|terrible|worst|take it back|keep it|away from me|no thanks)", re.I)
NAME = re.compile(r"\b(?i:my name is|i'm|i am|call me|name's)\s+([A-Z][a-z]+)")
NOT_NAMES = {"Not", "So", "Just", "Here", "Fine", "Good", "Sorry", "Back", "Tired", "Hungry", "Sad", "Happy", "Feeling", "Glad", "Okay", "Ok", "Really",
             "Going", "Looking", "From", "New", "Lost", "Afraid", "Scared", "The", "A", "An"}


def own_name(card):
    m = re.match(r"(?:You are|Name:)\s*([A-Z][a-z]+)", card or ""); return m.group(1) if m else ""


def broken(line, talk, card):
    """why a line is broken, or None"""
    line = line.strip(); me = own_name(card)
    words = re.findall(r"[A-Za-z']+", line)
    if len(words) <= 2 and not re.search(r"[.!?]", line[:-1] if line else ""): return "a name or a word alone"
    if talk and line.lower() == talk[-1].strip().lower(): return "echo"
    if len(talk) >= 2 and line.lower() == talk[-2].strip().lower(): return "repeats itself"
    if me and re.search(rf"(^|[,!?.]\s*){me}\b[!,?]|,\s*{me}[!.?]", line): return "calls someone by its own name"
    if re.search(r"walks up to|chat about the day|speaking first|exchanges, ", line): return "instructions leaked"
    if re.search(r"\b(game character|an ai|language model|npc|role-?play)\b", line, re.I): return "breaks character"
    return None


def gift_item(text):
    m = re.search(r"\b(?:here(?:'s| is)?|take|have|brought|got|give|giving|found)\b.*?\b(?:a|an|this|some|my)\s+([a-z ]{3,30}?)(?:[.!?,]|$| for you| i | you )", text, re.I)
    return m.group(1).strip().lower() if m else ""


def facts(line, talk, card):
    """why a line gets a fact wrong, or None (talk: the lines before it, the other side's last)"""
    prev = talk[-1] if talk else ""; low = line.lower()
    names = [n for t in talk[::-2][::-1] for n in NAME.findall(t) if n not in NOT_NAMES]   # the other side's lines
    if names and re.search(r"\b(remember|know|what'?s|what is|recall)\b.*\bname\b", prev, re.I) and names[-1].lower() not in low: return "forgot the player's name"
    hours = {m.group(1).strip().lower(): m.group(2) for m in re.finditer(r"([A-Za-z][A-Za-z ]+?) is open (\d+ am)", card or "")}
    shop = next((k for k in hours if k.split()[-1] in prev.lower()), None)
    if shop and re.search(r"\b(open|hours|close)", prev, re.I) and not re.search(rf"\b{hours[shop].split()[0]}\b", line): return "wrong opening time"
    lk = re.search(r"Likes: ([^.]*)\.", card or ""); dk = re.search(r"Dislikes: ([^.]*)\.", card or "")
    item = gift_item(prev)
    if item and re.search(r"\b(here|take|have|brought|got|give|giving|found|present|gift)\b", prev, re.I):
        likes = [x.strip().lower() for x in lk.group(1).split(",")] if lk else []; dislikes = [x.strip().lower() for x in dk.group(1).split(",")] if dk else []
        if any(d and (d in item or item in d) for d in dislikes) and POS.search(line) and not NEG.search(line): return "loves a gift it dislikes"
        if any(l and (l in item or item in l) for l in likes) and NEG.search(line) and not POS.search(line): return "hates a gift it likes"
    return None


def check(line, talk, card): return broken(line, talk, card) or facts(line, talk, card)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["filter", "score"]); ap.add_argument("--inp", required=True); ap.add_argument("--out", default="")
    A = ap.parse_args()
    if A.mode == "filter":
        why = collections.Counter(); kept = cut = dropped = 0
        with open(A.out, "w") as f:
            for l in open(A.inp):
                r = json.loads(l); ms = r["messages"]; card = next((m["content"] for m in ms if m["role"] == "system"), ""); out, talk = [], []
                for m in ms:
                    if m["role"] == "assistant":
                        w = check(m["content"], talk, card)
                        if w: why[w] += 1; break
                    out.append(m)
                    if m["role"] != "system": talk.append(m["content"])
                while out and out[-1]["role"] != "assistant": out.pop()
                if sum(m["role"] == "assistant" for m in out) < 2: dropped += 1; continue
                cut += len(out) < len(ms); kept += len(out) == len(ms); r["messages"] = out; f.write(json.dumps(r) + "\n")
        print(f"[rules] {kept} talks whole, {cut} cut, {dropped} dropped; reasons {dict(why.most_common())} -> {A.out}")
    else:
        by = collections.defaultdict(collections.Counter)
        for l in open(A.inp):
            r = json.loads(l); w = check(r["reply"], r["turns"], r["card"]); k = (r["model"], r.get("pick", ""))
            by[k]["n"] += 1; by[k]["ok"] += w is None
            if w: by[k][w] += 1
        for (m, p), c in sorted(by.items()):
            print(f"[rules] {m:<14}{p:<6} pass {100 * c['ok'] / c['n']:.1f}% of {c['n']}: " + ", ".join(f"{k} {v}" for k, v in c.most_common() if k not in ("n", "ok")))
