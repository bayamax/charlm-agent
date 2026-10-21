#!/usr/bin/env python3
"""Longer cards for chat training: a game hands the model a card full of world and state lines, most of them beside the
point of what the player just said. This adds such lines (made from templates) to the card of chat conversations, so
the model learns to answer the player and use what fits, not to lose its way in a long card.

  python3 card_pad.py --in chat.jsonl --out chat_long.jsonl --frac 0.6 --min 3 --max 10
  python3 card_pad.py --in npc_val.jsonl --out npc_val_long.jsonl --frac 1 --min 10 --max 10 --bank test   (other lines, for scoring)
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.6, help="share of the conversations with a card that get lines added")
ap.add_argument("--min", type=int, default=3); ap.add_argument("--max", type=int, default=10)
ap.add_argument("--bank", default="train", choices=["train", "test"]); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
R = random.Random(A.seed)

PLACES = ["the well", "the mill", "the old bridge", "the chapel", "the inn", "the market square", "the harbor", "the north gate",
          "the bakery", "the forge", "the library", "the orchard", "the watchtower", "the stables"]
THINGS = ["bread", "rope", "candles", "salt", "iron nails", "wool", "lamp oil", "dried fish", "apples", "cloth", "herbs", "firewood"]
WEATHER = ["It is raining lightly.", "The sky is clear.", "A cold wind is blowing.", "It is a warm afternoon.", "Fog hangs over the fields.",
           "Snow covers the rooftops.", "It is late at night.", "The sun is about to set."]
TRAIN = [
    lambda: f"{R.choice(PLACES).capitalize()} is {R.choice(['north', 'south', 'east', 'west'])} of here.",
    lambda: f"You have {R.randint(0, 40)} gold.",
    lambda: f"You feel {R.choice(['tired', 'rested', 'hungry', 'cheerful', 'a little bored', 'busy'])}.",
    lambda: R.choice(WEATHER),
    lambda: f"Today you {R.choice(['sold', 'bought', 'carried', 'counted', 'mended'])} some {R.choice(THINGS)}.",
    lambda: f"{R.choice(THINGS).capitalize()} cost{'' if R.random() < 0.5 else 's'} {R.randint(1, 9)} coins at the market.",
    lambda: f"The {R.choice(['mayor', 'priest', 'captain', 'miller', 'old healer', 'tax collector'])} lives near {R.choice(PLACES)}.",
    lambda: f"{R.choice(PLACES).capitalize()} {R.choice(['closes at dusk', 'opens at dawn', 'is busy today', 'was rebuilt last year', 'is quiet this week'])}.",
    lambda: f"Your health is {R.randint(1, 10)} out of 10.",
    lambda: f"You are carrying {R.randint(1, 6)} {R.choice(THINGS)}.",
    lambda: f"It is day {R.randint(2, 30)} of the {R.choice(['harvest', 'spring', 'winter', 'summer'])} season.",
    lambda: f"A {R.choice(['merchant', 'soldier', 'pilgrim', 'bard', 'stranger'])} arrived {R.choice(['this morning', 'last night', 'an hour ago'])}.",
]
TEST = ["The well in the square was dug long ago.", "Market day is every third morning.", "A cold wind blows from the hills in winter.",
        "The mill by the river grinds wheat for the whole valley.", "Children play near the old oak tree.", "The road north leads to the city of Varn.",
        "Candles are sold at the chandler's shop.", "The bridge was repaired last spring.", "Sheep graze on the eastern meadow.", "The bell tower rings at noon.",
        "Your boots are worn through at the heel.", "There are 12 coins in your purse."]


def lines(k):
    if A.bank == "test": return R.sample(TEST, min(k, len(TEST)))
    out = []
    while len(out) < k:
        s = R.choice(TRAIN)()
        if s not in out: out.append(s)
    return out


n = padded = 0
with open(A.out, "w") as f:
    for l in open(A.inp):
        r = json.loads(l); n += 1
        sys_ = next((m for m in r["messages"] if m["role"] == "system"), None)
        if sys_ is not None and R.random() < A.frac:
            add = " ".join(lines(R.randint(A.min, A.max))); padded += 1
            sys_["content"] = sys_["content"].rstrip() + ("\n" if "\n" in sys_["content"] else " ") + add
        f.write(json.dumps(r) + "\n")
print(f"[pad] {padded} of {n} conversations got longer cards -> {A.out}")
