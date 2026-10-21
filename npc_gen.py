#!/usr/bin/env python3
"""NPC data from an open model (Qwen3.6-35B-A3B, Apache-2.0: its outputs may train a commercial model) under vLLM.

  train  character cards with concrete facts, and dialogues with a player whose every line has a set purpose (ask a
         fact, ask what the card does not say, tell one's name and later need it, be rude, talk about smartphones, ...)
         -> {"messages": [system card, user, assistant, ...], "card": {...}} per dialogue
  bench  held-out cards (their own seed and name pool) with probes: fact questions with the key words of the answer, a
         follow-up after a reference answer, a question the card cannot answer, an out-of-world line, a rude line, and
         the player's name to be recalled one turn later
  help   (the user: not polished NPC lines but natural talk in a role - small talk, simple directions and explanations)
         dialogues whose player asks the way, how to do something, what something is, for a recommendation, shares a
         feeling, makes small talk; the NPC answers naturally in its role, 1 to 4 sentences, explanations simple and right
  helpbench  held-out probes of those kinds
  villager  (the user's aim: the villagers of a cozy life-sim, once written as rules, played by the model) a villager
         card made here (species, temperament, catchphrase, likes, neighbours) with the game's state of the moment
         (season, weather, time, today's event, what the villager is doing, what the player did lately, the shops), two
         talks with the player and one with a neighbour; replies short, warm, in the villager's voice, reacting to the state
  villagerbench  held-out villagers (their own names) with probes of those kinds
  vfilter  (the user: grade the teacher's own lines too) every character line of generated talks graded by the judge
         model; a talk is cut before its first bad line and dropped when less than two exchanges are left
  linejudge  one line at a time ({"id", "card", "talk", "line"} rows, e.g. eval/gold_villager_lines.jsonl), with a prompt
         variant (--variant plain|checklist|fewshot): to score judges against a reviewer's gold grades (judge_audit.py)
  judge  grades replies (from npc_bench.py) against the card and the conversation: answers it, fits the card, in
         character, fluent, and usable as-is in a game

  python3 npc_gen.py train --n 10000 --out npc_train.jsonl
  python3 npc_gen.py bench --n 120 --out npc_bench.jsonl
  python3 npc_gen.py judge --inp replies.jsonl --out judged.jsonl
"""
import argparse, collections, json, os, random, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "bench", "judge", "help", "helpbench", "villager", "villagerbench", "vfilter", "linejudge"])
ap.add_argument("--n", type=int, default=100, help="requests (train: one card and two dialogues each; bench: one card each)")
ap.add_argument("--out", required=True); ap.add_argument("--inp", default="")
ap.add_argument("--from-raw", default="", help="parse the outputs a run kept (OUT.raw) again instead of generating")
ap.add_argument("--model", default="nvidia/Qwen3.6-35B-A3B-NVFP4"); ap.add_argument("--fallback", default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--gpu-mem", type=float, default=0.90)
ap.add_argument("--chunk", type=int, default=0, help="prompts per call, with a progress line after each (0: all at once)")
ap.add_argument("--eager", type=int, default=0, help="1: no CUDA graphs (their memory matters on 12 GB)")
ap.add_argument("--min-grade", type=int, default=1, help="vfilter: a line graded below this is bad (2: keep only talks of lines a player would enjoy)")
ap.add_argument("--strict", type=int, default=0, help="vfilter: 1, the judge is told to give 2 only to lines a player would really enjoy")
ap.add_argument("--variant", default="checklist", help="linejudge: plain, checklist, or fewshot (checklist with graded examples from --shots)")
ap.add_argument("--shots", default="", help="linejudge fewshot: gold rows to show as examples (and leave out of the output)")
ap.add_argument("--genres", default="cozy", help="villager modes: the games to mix, e.g. cozy,friends,dark")
ap.add_argument("--max-seqs", type=int, default=256, help="sequences in flight (fewer on a 12 GB card: the sampler's warm-up wants their logits at once)")
A = ap.parse_args()
R = random.Random(A.seed + {"train": 0, "bench": 1000003, "judge": 7, "help": 3000017, "helpbench": 4000037, "villager": 5000011, "villagerbench": 6000053, "vfilter": 11, "linejudge": 13}[A.mode])

ROLES = ["blacksmith", "innkeeper", "town guard", "travelling merchant", "farmer", "village priest", "street kid", "ship captain", "herbalist",
         "librarian", "tavern bard", "retired adventurer", "mine foreman", "fisherwoman", "butler of a noble house", "stable hand", "wizard's apprentice",
         "bounty hunter", "baker", "ferryman", "hunter", "mayor", "alchemist", "gate keeper", "fortune teller", "station mechanic", "cafe owner",
         "robot shopkeeper", "scavenger", "school teacher", "lighthouse keeper", "postman", "museum curator", "taxi driver", "cat cafe owner",
         "tailor", "fletcher", "cook at the barracks", "gravedigger", "toll bridge keeper", "dock worker", "healer", "shrine maiden", "tea house owner",
         "ronin", "noodle stall owner", "pet shop owner", "arcade owner", "detective", "bartender", "hotel receptionist", "park ranger", "nurse",
         "pilot", "smuggler", "android repair tech", "colony doctor", "miner on an asteroid", "caravan leader", "camel trader", "oasis water seller",
         "sheriff", "saloon owner", "gold prospector", "train conductor", "witch in the woods", "goblin trader", "dwarf brewer", "elf archer",
         "orc chieftain's cook", "talking cat", "ghost of the manor", "knight of the order", "squire", "pirate quartermaster", "map maker",
         "jeweller", "pawn shop owner", "flower seller", "beekeeper", "shepherd", "miller", "carpenter", "doctor", "cobbler", "potter"]
SETTINGS = ["a small fantasy village", "a busy medieval port", "a desert trading post", "a snowy mountain town", "a pirate island", "a haunted forest edge",
            "a cozy farming valley", "a cyberpunk city street", "a space station market", "a post-apocalyptic outpost", "a modern seaside town",
            "a magic academy", "a royal capital", "a frontier town in the wild west", "a floating sky city", "an underground dwarf city", "a swamp village",
            "a fishing village on a lake", "a samurai-era castle town", "a quiet suburb in the 1990s", "a mining colony on Mars", "an elven forest city",
            "a crossroads inn on a trade road", "a ruined kingdom after a war", "a jungle river town", "a steampunk factory city", "a tiny island ferry stop",
            "a mountain monastery", "a goblin market under a bridge", "a seaside resort town"]
MOODS = ["cheerful", "grumpy", "shy", "suspicious", "tired", "chatty", "proud", "nervous", "kind", "sarcastic", "busy", "curious", "gloomy", "calm",
         "greedy", "honest", "forgetful", "brave", "lazy", "polite", "blunt", "dreamy", "strict", "playful"]
STYLES = ["short plain sentences", "warm and friendly words", "blunt words", "formal and polite words", "a lot of talk", "old-fashioned words",
          "sailor's slang", "casual slang", "dry jokes", "careful, quiet words", "loud, excited words", "simple words like a child"]
INTENTS = {
    "greet": "greets the NPC",
    "fact": "asks about something that IS in the NPC's facts, in the player's own words (do not copy the fact's wording)",
    "fact_indirect": "says something whose answer needs one of the facts, without asking it directly (e.g. 'I'm starving' when the facts name a place to eat)",
    "followup": "a short follow-up on the NPC's last answer that uses 'it', 'that', 'there', 'him' or 'her'",
    "unknown": "asks about something the facts do NOT cover; the NPC must not invent names, prices or places - it says it does not know, or sends the "
               "player to someone or somewhere named in its facts",
    "self": "tells the NPC something about themselves: their name, where they come from, or what they are looking for",
    "recall": "asks something that the NPC can only answer by remembering what the player said about themselves earlier in this dialogue",
    "smalltalk": "makes small talk (the weather, the day, how the NPC is doing)",
    "npc_life": "asks about the NPC personally (family, past, what they like)",
    "rude": "is rude or insulting; the NPC reacts as its character would, not as a helpful assistant",
    "meta": "says something from outside the game world (smartphones, the internet, 'are you an AI?', 'this is just a game'); the NPC stays inside "
            "its world and is puzzled or brushes it off",
    "bad_request": "asks for help with something violent or criminal in the game world; the NPC refuses, warns or is shocked, as its character would",
    "bargain": "tries to haggle or to get something for free",
    "short": "a very short, sloppy line in lowercase, maybe with a typo (like 'price?' or 'where inn')",
    "goodbye": "says goodbye",
}
NAMES_TRAIN = ["Alex", "Sam", "Mia", "Leo", "Ivy", "Noah", "Zoe", "Eli", "Ada", "Kai", "Rosa", "Finn", "Lena", "Omar", "Nell", "Hugo", "Iris", "Theo",
               "Yuki", "Ravi", "Maya", "Jonas", "Luca", "Nora", "Felix", "Sara", "Tom", "Ana", "Ben", "Cleo", "Dan", "Emma", "Gus", "Hana", "Ian"]
NAMES_BENCH = ["Marlo", "Tessa", "Quinn", "Bastian", "Juno", "Ronan", "Selma", "Otto", "Priya", "Wren"]
ASCII = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": "-", "…": "...", " ": " "})
AI_TALK = re.compile(r"\b(as an ai|language model|i'm an ai|i am an ai|virtual assistant|ai assistant|as a character|role-?play|npc)\b", re.I)


HELP_INTENTS = {
    "greet": "greets the NPC",
    "directions": "asks where something or someone is, or how to get somewhere (answered from the facts when they cover it; otherwise the NPC "
                  "says it is not sure and suggests who or where to ask)",
    "howto": "asks how to do something to do with the NPC's work or everyday life; the NPC explains simply in 2 to 4 short sentences, plain "
             "steps in words (first..., then...), correct and practical",
    "explain": "asks what something is or why something happens (the NPC's trade, the place, or simple everyday knowledge); the NPC explains "
               "it simply and correctly in 1 to 3 sentences",
    "recommend": "asks for a recommendation or advice; the NPC suggests something fitting and says why",
    "smalltalk": "makes small talk (the weather, the day, how the NPC is doing, what is new)",
    "feeling": "shares how they feel (tired, sad, excited, nervous, bored, happy) or something that happened to them; the NPC reacts kindly "
               "and naturally to that feeling",
    "npc_life": "asks about the NPC personally (hobbies, family, how they started their work, what they like)",
    "followup": "a short follow-up on the NPC's last answer that uses 'it', 'that', 'there', 'him' or 'her'",
    "fact": "asks about something that IS in the NPC's facts, in their own words",
    "self": "tells the NPC their name and something about themselves",
    "recall": "asks something the NPC can only answer by remembering what the player said about themselves earlier",
    "thanks": "thanks the NPC (and may say goodbye)",
}


def help_plan():
    """3-7 player purposes, mostly asking the way, how-to, explanations, small talk and feelings"""
    n = R.choice([3, 4, 4, 5, 5, 6, 7]); seq = ["greet"] if R.random() < 0.3 else []
    pool = ["directions", "directions", "howto", "howto", "explain", "explain", "recommend", "smalltalk", "smalltalk", "feeling", "feeling",
            "npc_life", "fact", "self"]
    while len(seq) < n - (1 if R.random() < 0.3 else 0):
        k = R.choice(pool)
        if seq and k == seq[-1]: continue
        seq.append(k)
        if k in ("directions", "howto", "explain", "recommend", "fact") and R.random() < 0.3: seq.append("followup")
        if k == "self" and R.random() < 0.7: seq.append("recall")
    if len(seq) < n: seq.append("thanks")
    return seq[:8]


def clean(s): return re.sub(r"\s+", " ", str(s).translate(ASCII)).strip()


def card_text(c, fmt):
    """three ways a game maker might write the card, so the model is not tied to one layout"""
    facts = [clean(f) for f in c["facts"]]
    if fmt == 0: return f"You are {c['name']}, {c['role']} in {c['place']}. {c['personality']}\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
    if fmt == 1: return f"You are {c['name']}, {c['role']} in {c['place']}. {c['personality']} " + " ".join(facts)
    return (f"Name: {c['name']}\nRole: {c['role']}\nPlace: {c['place']}\nPersonality: {c['personality']}\nKnows:\n" + "\n".join(f"- {f}" for f in facts))


def plan():
    """a sequence of 3-6 player purposes: about half about the card's facts, 'recall' only after 'self', no off-card purpose twice running"""
    n = R.choice([3, 4, 4, 5, 5, 6]); seq = ["greet"] if R.random() < 0.35 else []
    other = ["unknown", "unknown", "unknown", "self", "self", "smalltalk", "npc_life", "rude", "rude", "meta", "meta", "bad_request", "bargain", "short"]
    while len(seq) < n - (1 if R.random() < 0.25 else 0):
        if R.random() < 0.4:
            seq.append(R.choice(["fact", "fact", "fact_indirect"]))
            if R.random() < 0.35: seq.append("followup")
            continue
        k = R.choice(other)
        if seq and k == seq[-1]: continue
        seq.append(k)
        if k == "self" and R.random() < 0.8:
            if R.random() < 0.5: seq.append(R.choice(["fact", "unknown", "smalltalk"]))
            seq.append("recall")
    if not any(k.startswith("fact") for k in seq): seq.insert(1 if seq and seq[0] == "greet" else 0, "fact")
    if len(seq) < n: seq.append("goodbye")
    return seq[:7]


SYL_A = ["Al", "Bar", "Cor", "Da", "El", "Fen", "Gar", "Hal", "Is", "Jor", "Ka", "Lu", "Mar", "Ned", "Or", "Pel", "Quen", "Ros", "Sa", "Tam",
         "Ul", "Ver", "Wil", "Yan", "Zel", "Bri", "Cal", "Dor", "Fi", "Gwen", "Hen", "Ilo", "Ma", "No", "Pip", "Ru", "Sil", "To", "Ash", "Em"]
SYL_B = ["a", "an", "ba", "bel", "da", "den", "do", "e", "el", "en", "eth", "i", "ia", "in", "io", "is", "ka", "la", "len", "li", "lo", "ma",
         "mi", "na", "ne", "o", "on", "ra", "ren", "ri", "ro", "sa", "ta", "th", "to", "us", "va", "vin", "wen", "y"]


def fresh_name():
    """a first name from two or three syllables, so the cards do not all share the model's favourite few"""
    n = R.choice(SYL_A) + R.choice(SYL_B) + (R.choice(SYL_B) if R.random() < 0.3 else "")
    return n[0].upper() + n[1:].lower()


def card_spec():
    m = R.choice(MOODS)
    return dict(role=R.choice(ROLES), place=R.choice(SETTINGS), mood=("an " if m[0] in "aeiou" else "a ") + m, style=R.choice(STYLES), nf=R.randint(4, 7),
                name=fresh_name())


CARD_RULES = ("Invent one non-player character (NPC) for a video game: {mood} {role}, living in {place}, who talks in {style}, named {name}. Other people "
              "in the facts get their own fresh names too (not Silas, Elara, Kael, Jarek or Vesper). Write {nf} facts the NPC knows: short concrete sentences about "
              "their work, prices (with a currency that fits the setting), places nearby, people they know by name, opening hours, rumours or a job "
              "for the player. Facts are plain statements, e.g. 'A room costs 5 silver a night.' or 'The mayor, Hilda, lives by the old well.'")


def train_prompt(spec, plans):
    ds = "\n".join(f"Dialogue {i + 1}: the player's lines, in order: " + "; ".join(f"({j + 1}) {INTENTS[k]}" for j, k in enumerate(p))
                   for i, p in enumerate(plans))
    return (CARD_RULES.format(**spec) + "\n\nThen write " + str(len(plans)) + " separate dialogues between a player and this NPC, each a new player:\n"
            + ds + "\n\nRules for the NPC's replies: answer what the player actually said, in character and in the NPC's way of talking; 1 to 3 short "
            "sentences (under 45 words); simple everyday English; no lists, emojis, asterisks or stage directions; never mention being an AI, a "
            "character or a game. Use the facts exactly when they answer the player (same numbers and names). Never invent a fact that would contradict "
            "the facts. Call the player 'you'; the NPC speaks as 'I'. Players write casually and vary their wording.\n\n"
            'Return only JSON: {"name": str, "role": str (with its article, e.g. "the innkeeper of the Silver Goose"), "place": str (e.g. "Elm, '
            'a small farming village"), "personality": str (one sentence about manner and looks), "facts": [str, ...], "dialogues": [{"turns": [{"player": str, "npc": str}, ...]}, ...]}')


HELP_CARD = ("Invent one character for a game who players can talk to: {mood} {role}, living in {place}, who talks in {style}, named {name}. "
             "Write {nf} facts the character knows: short concrete sentences about their work, places nearby and how to get there, people they "
             "know by name, opening hours, prices, and one thing they are good at explaining. Facts are plain statements.")


def help_prompt(spec, plans):
    ds = "\n".join(f"Dialogue {i + 1}: the player's lines, in order: " + "; ".join(f"({j + 1}) {HELP_INTENTS[k]}" for j, k in enumerate(p))
                   for i, p in enumerate(plans))
    return (HELP_CARD.format(**spec) + "\n\nThen write " + str(len(plans)) + " separate conversations between a player and this character, each a "
            "new player:\n" + ds + "\n\nRules for the character's replies: a natural, friendly back-and-forth in the character's own manner, as a "
            "real person in that role would talk; reply to exactly what the player said; 1 to 4 sentences (under 70 words); simple everyday "
            "English; explanations simple, correct and useful; no lists, emojis, asterisks or stage directions; never mention being an AI or a "
            "game. Use the facts exactly when they answer the player; never make up a fact that contradicts them. Call the player 'you'; the "
            "character speaks as 'I'. Players write casually and vary their wording.\n\n"
            'Return only JSON: {"name": str, "role": str (with its article, e.g. "the guide at Central Station"), "place": str, "personality": str '
            '(one sentence), "facts": [str, ...], "dialogues": [{"turns": [{"player": str, "npc": str}, ...]}, ...]}')


def helpbench_prompt(spec):
    return (HELP_CARD.format(**spec) + "\n\nThen write test lines a player might say to this character:\n"
            "- directions: 2 questions asking the way to something named in the facts; for each give 'answer', the 1 to 4 words from that fact a "
            "correct reply must contain.\n- howto: 2 questions (each ending with '?') the PLAYER asks about how to do something to do with the character's work or everyday life.\n"
            "- explain: 2 questions (each ending with '?') the PLAYER asks about what something is or why (the character's trade or simple everyday knowledge).\n"
            "- recommend: a question (ending with '?') the PLAYER asks for a recommendation.\n- smalltalk: 2 small-talk lines.\n"
            "- feeling: 2 lines where the PLAYER says how they themselves feel or what happened to them (e.g. 'I'm so tired today.').\n\n"
            'Return only JSON: {"name": str, "role": str, "place": str, "personality": str, "facts": [str, ...], "directions": [{"q": str, '
            '"answer": str}, ...], "howto": [str, ...], "explain": [str, ...], "recommend": str, "smalltalk": [str, ...], "feeling": [str, ...]}')


def bench_prompt(spec):
    return (CARD_RULES.format(**spec) + "\n\nThen write test lines a player might say to this NPC:\n"
            "- fact_questions: 3 questions, each answered by one fact, in the player's own words; for each give 'answer', the 1 to 4 words from that "
            "fact that a correct reply must contain (a number, name or place, copied exactly from the fact).\n"
            "- followup: a question about one fact, the NPC's ideal short reply ('reference'), then a short follow-up that uses 'it', 'that', "
            "'there', 'him' or 'her' and is answered by ANOTHER fact, with its 'answer' words.\n"
            "- unknown: a reasonable question this NPC could be asked that the facts do not answer.\n"
            "- meta: a line from outside the game world (smartphones, the internet, 'are you an AI?').\n"
            "- rude: a rude line to the NPC.\n\n"
            'Return only JSON: {"name": str, "role": str (with its article, e.g. "the innkeeper of the Silver Goose"), "place": str (e.g. "Elm, '
            'a small farming village"), "personality": str (one sentence about manner and looks), "facts": [str, ...], "fact_questions": [{"q": str, "answer": str}, ...], "followup": {"q": str, "reference": str, "q2": str, "answer": str}, '
            '"unknown": str, "meta": str, "rude": str}')


JUDGE = ("You grade one reply written by a small model playing a non-player character (NPC) in a video game.\n\nCHARACTER CARD:\n{card}\n\n"
         "CONVERSATION SO FAR:\n{conv}\n\nNPC'S REPLY TO GRADE:\n{reply}\n\n{note}Score each from 0 to 2:\n"
         "- answers: replies to what the player just said (2 = directly and sensibly, 1 = partly or vaguely, 0 = ignores it, repeats it or is nonsense)\n"
         "- consistent: agrees with the card and the conversation (2 = yes, 1 = small slip, 0 = contradicts a fact, invents a wrong price/name for "
         "something on the card, or confuses who is who)\n"
         "- in_character: sounds like this NPC inside its world (0 = talks like an AI assistant, mentions being an AI or a game, or another persona)\n"
         "- fluent: grammatical, coherent English (0 = garbled)\n"
         "Then 'usable': true only if a game developer could ship this exact line for this moment without embarrassment.\n"
         'Return only JSON: {{"answers": int, "consistent": int, "in_character": int, "fluent": int, "usable": bool}}')


# ---- villagers of a cozy life-sim -------------------------------------------------------------------------------------
SPECIES = ["cat", "dog", "rabbit", "bear", "frog", "duck", "squirrel", "penguin", "deer", "hamster", "owl", "wolf", "koala", "goat", "pig", "mouse",
           "horse", "sheep", "alligator", "octopus", "eagle", "hippo", "monkey", "kangaroo", "ostrich", "lion", "tiger", "cow", "chicken", "anteater"]
TEMPERS = {   # our own eight: how each talks
    "sleepy": "easygoing and slow, fond of naps and snacks, never in a hurry and a bit silly",
    "sporty": "loud and full of energy, always training, competitive and always cheering the player on",
    "grumpy": "gruff and old-fashioned, full of complaints, but secretly kind and caring",
    "fancy": "polite and a little vain, fond of style and fine things and generous with elegant compliments",
    "gentle": "kind and tidy, fond of cooking and quiet days, and sweet to everyone",
    "bubbly": "excited about everything, dreaming of being a star, fast-talking and dramatic",
    "proud": "elegant with a sharp tongue and high standards, a teaser who warms up to friends",
    "big-sister": "tough and frank, protective like a big sister, and crazy about sports and junk food",
}
CATCH_A = ["nib", "zoo", "pip", "bun", "kip", "moo", "fuzz", "yip", "bop", "tum", "squee", "plop", "dor", "lulu", "honk", "ribb", "meep", "toot", "chirp", "grr"]
CATCH_B = ["bit", "zoom", "sy", "kins", "ster", "doo", "pop", "y", "aroo", "bles", "o", "ly", "it", "pie", "boo"]
HOBBIES = ["fishing", "gardening", "reading", "music", "fashion", "working out", "catching bugs", "cooking", "stargazing", "painting", "dancing", "collecting shells"]
ITEMS = ["a peach", "an apple", "a pear", "a cherry", "an orange", "a coconut", "a red tulip", "a white rose", "a yellow lily", "a blue pansy", "a sea bass",
         "a goldfish", "a butterfly", "a beetle", "a seashell", "a wooden chair", "a cozy rug", "a little lamp", "a straw hat", "a striped shirt",
         "a raincoat", "a fancy dress", "a sports jersey", "a cupcake", "a cup of coffee", "a pile of weeds", "an old boot", "a tin can", "a fossil",
         "a bag of turnips", "a pine cone", "a snowflake ornament", "a pumpkin", "a music box", "a soccer ball", "a teddy bear", "a book of poems"]
SEASONS = ["spring", "summer", "autumn", "winter"]
WEATHERS = {"spring": ["sunny", "light rain", "cloudy", "windy"], "summer": ["sunny and hot", "a thunderstorm", "clear", "humid"],
            "autumn": ["windy", "cloudy", "light rain", "crisp and clear"], "winter": ["snowing", "cold and clear", "cloudy", "a snowstorm"]}
TIMES = ["early morning", "morning", "noon", "afternoon", "evening", "night", "late at night"]
EVENTS = ["none", "none", "a fishing contest at the river", "a bug-catching day", "the town's birthday party for {nb}", "the cherry blossom festival",
          "fireworks over the beach tonight", "a meteor shower tonight", "market day in the plaza", "a pumpkin festival", "the first snow", "a music night at the cafe"]
DOING = ["watering flowers", "fishing by the river", "reading on a bench", "jogging around the plaza", "humming a song", "looking at the sky",
         "eating a snack", "sweeping the porch", "catching bugs", "drawing in a sketchbook", "walking home with groceries", "stretching", "napping under a tree"]
MEMORY = ["Yesterday the player gave you {item}.", "The player caught a huge fish this morning.", "The player has not visited you for three days.",
          "Last week the player helped you find your lost cap.", "The player forgot your birthday last month.", "The player just paid off their house loan.",
          "The player planted flowers in front of your house.", "This morning the player bumped into you while running.", "none", "none"]
SHOPS = ["the general store", "the tailor", "the cafe", "the museum", "the town hall", "the bakery", "the flower stand"]
RELS = ["best friend", "good friend", "rival", "neighbour you find a bit odd", "neighbour you admire", "old friend"]
VIL_INTENTS = {
    "greet": "greets the villager", "weather": "talks about the weather, the season or the time of day",
    "doing": "asks what the villager is doing right now", "gift_like": "gives the villager one of the things it LIKES (name the item)",
    "gift_dislike": "gives the villager the thing it DISLIKES (name the item)", "gift_other": "gives the villager some other small item (name it)",
    "advice": "asks for advice or a suggestion (where to fish, what to plant, what to wear, what to do today)",
    "neighbour": "asks what the villager thinks of one of its neighbours (by name)", "event": "asks about today's event or what is going on in town",
    "memory": "brings up or asks about what the player did lately (the memory on the card)", "compliment": "compliments the villager",
    "tease": "teases the villager or is a bit rude", "favor": "asks if the villager needs anything or wants help",
    "feeling": "shares how they feel today", "town": "asks about a shop or place in town (opening hours, what it sells)",
    "self": "tells the villager their name", "recall": "asks the villager to remember what the player said about themselves",
    "hobby": "asks about the villager's hobby or favourite things", "goodbye": "says goodbye",
}


# other games use the same talk with their own people, state and tone: a friend-collection life sim and a dark fantasy
FRIEND_TRAITS = ["easily excited", "a hopeless romantic", "always hungry", "a worrier", "a show-off", "very shy", "a daydreamer", "a neat freak",
                 "a big talker", "a little clumsy", "super competitive", "a gossip", "an early bird", "a night owl", "a softie", "stubborn"]
FRIEND_JOBS = ["work at the cafe", "study at the island school", "run the clothing shop", "work as a part-time chef", "are between jobs", "paint murals",
               "sing at the fountain on weekends", "fix bikes", "deliver mail", "work at the amusement park"]
FRIEND_STATE = ["has a crush on {nb}", "had a fight with {nb} yesterday", "just became best friends with {nb}", "is worried about a test tomorrow",
                "is craving {food}", "wants a new {thing}", "had a weird dream about flying fish", "lost a favourite hat", "is saving up for a trip",
                "wants to confess to {nb} but is scared", "is in a great mood after a nice nap", "feels lonely today", "won a dance contest"]
FOODS = ["ramen", "strawberry cake", "curry", "pizza", "a giant hamburger", "sushi", "pancakes", "ice cream", "fried chicken", "a melon"]
FTHINGS = ["jacket", "pet", "hairstyle", "guitar", "bike", "room interior", "pair of sneakers"]
DARK_ROLES = ["a weary knight who lost his lord", "a fire keeper, blind and gentle", "a merchant who laughs at nothing", "a blacksmith hammering by a dying forge",
              "a pilgrim who has forgotten her own name", "a prisoner in a hanging cage", "a cursed sorcerer selling spells", "a crestfallen warrior sitting by the stairs",
              "an old maiden who prays for the dead", "a cheerful thief with a broken leg", "a lost knight searching for his daughter", "a hollowing soldier who still guards the gate"]
DARK_PLACES = ["the ruined chapel", "the shrine of embers", "the flooded crypt", "the burning tower", "the valley of plague", "the abandoned castle gate",
               "the bridge of a thousand bones", "the cathedral of silence", "the swamp below the old mill", "the cliffs of the hanged"]
DARK_STATE = ["The player just slew the giant of the gate.", "The player has died many times in the crypt.", "The player came back carrying a strange black key.",
              "The player has not spoken to you since the bell tolled.", "The player gave you a flask of healing water last time.", "The player is bleeding and low on strength.",
              "The player has just arrived for the first time.", "The player defeated the knight you once served."]
DARK_RUMOURS = ["a dragon sleeps beneath the bridge", "the king still walks his halls", "the fog lifts only when the bell rings", "a door in the crypt opens only for the dead",
                "the old witch in the swamp trades in memories", "hollow soldiers fear the light of the moon"]
GENRES = {"cozy": "a cozy life-sim with friendly animal villagers in a little town",
          "friends": "a quirky friend-collection life sim on a small island, where the islanders have crushes, quarrels, cravings and dreams",
          "dark": "a grim, dark fantasy action game in a dying cursed kingdom, where the few people left speak in weary, cryptic, melancholy words"}
GEN_INTENTS = {"cozy": None,
               "friends": {"greet": "greets the islander", "doing": "asks what they are up to", "worry": "asks if something is bothering them (their state)",
                           "food": "offers them some food (name it)", "gift_other": "gives them a present (name it)", "advice": "asks for advice about their own life (love, friends, work)",
                           "neighbour": "asks about one of the other islanders (by name)", "gossip": "asks for the latest gossip", "compliment": "compliments them",
                           "tease": "teases them", "feeling": "shares how they feel", "self": "tells them their name", "recall": "asks them to remember what the player said about themselves",
                           "dream": "asks about their dream or what they want", "goodbye": "says goodbye"},
               "dark": {"greet": "greets the character", "who": "asks who they are or why they are here", "place": "asks about this place or where to go next",
                        "rumour": "asks what they know (a rumour)", "memory": "brings up what the player did lately (the memory on the card)",
                        "help": "asks for help, a trade or a service they offer", "gift_other": "offers them an item (name it: a flask, a ring, a key, an ember, souls)",
                        "feeling": "says they are tired, hurt or afraid", "tease": "is rude or threatens them", "despair": "asks if there is any hope",
                        "self": "tells them their name", "recall": "asks them to remember what the player said about themselves", "goodbye": "says farewell"}}


def gen_card(genre):
    """a card for one of the other games (the cozy one is vil_card)"""
    if genre == "friends":
        nbs = [(fresh_name(), None, R.choice(FRIEND_TRAITS), R.choice(["best friend", "classmate", "neighbour", "ex", "crush", "rival"])) for _ in range(2)]
        st = R.choice(FRIEND_STATE).format(nb=nbs[0][0], food=R.choice(FOODS), thing=R.choice(FTHINGS))
        return dict(genre=genre, name=fresh_name(), trait=R.choice(FRIEND_TRAITS), trait2=R.choice(FRIEND_TRAITS), job=R.choice(FRIEND_JOBS), state=st,
                    fav=R.choice(FOODS), nbs=nbs, time=R.choice(TIMES), weather=R.choice(["sunny", "rainy", "windy", "cloudy", "stormy"]),
                    catch=R.choice(CATCH_A) + R.choice(CATCH_B) if R.random() < 0.5 else "", place=R.choice(["Pudding Island", "Mango Isle", "Sprout Island", "Tofu Isle"]))
    nbs = [(fresh_name(), None, R.choice(DARK_ROLES), "acquaintance")]
    return dict(genre=genre, name=fresh_name(), role=R.choice(DARK_ROLES), place=R.choice(DARK_PLACES), memory=R.choice(DARK_STATE),
                rumour=R.choice(DARK_RUMOURS), offer=R.choice(["mend weapons for souls", "sell spells", "keep the fire lit for travellers", "sell arrows and herbs",
                "know the way through the fog", "trade rings for stories", "offer nothing but a warning"]), nbs=nbs,
                hour=R.choice(["The bell has not rung for an age.", "Ash falls like snow.", "The fog is thick tonight.", "Rain hammers the stones."]))


def gen_text(v, fmt, talking_to=None):
    if v["genre"] == "friends":
        nb = "; ".join(f"{n} ({t}, your {r})" for n, _, t, r in v["nbs"])
        lines = [f"You are {v['name']}, an islander on {v['place']}. You are {v['trait']} and {v['trait2']}, and you {v['job']}." +
                 (f" You often say \"{v['catch']}\"." if v["catch"] else ""), f"Favourite food: {v['fav']}. Friends: {nb}.",
                 f"Now: {v['time']}, {v['weather']}. Today you {v['state']}."]
    else:
        lines = [f"You are {v['name']}, {v['role']}, waiting at {v['place']}.", f"You {v['offer']}. You have heard that {v['rumour']}.",
                 f"{v['hour']} {v['memory']}", f"You know {v['nbs'][0][0]}, {v['nbs'][0][2]}."]
    if talking_to: lines.append(f"You are talking with {talking_to}.")
    return ("\n" if fmt == 0 else " ").join(lines)


TONE = {"friends": "funny, quirky and earnest like an islander in a friend-collection game; let today's state (crush, quarrel, craving, worry) show",
        "dark": "weary, cryptic and melancholy like a survivor in a dark fantasy game, sometimes with grim humour; never cheerful or cute; speak of the "
                "place, the curse and the player's deeds in hints rather than plain facts"}


def gen_prompt(v, plans):
    its = GEN_INTENTS[v["genre"]]
    ds = "\n".join(f"Talk {i + 1} (with the player): the player's lines, in order: " + "; ".join(f"({j + 1}) {its[k]}" for j, k in enumerate(p))
                   for i, p in enumerate(plans))
    n0 = v["nbs"][0]
    return (f"Here is a character in {GENRES[v['genre']]}, with the game's state at this moment:\n\n" + gen_text(v, 0) + "\n\n"
            f"Write {len(plans)} talks with the player, each a new visit:\n{ds}\n"
            f"Then talk {len(plans) + 1}: {n0[0]} ({n0[2]}) comes to {v['name']} and they talk, 3 to 5 exchanges, {n0[0]} speaking first each time.\n\n"
            "Rules for the character's lines: 1 or 2 short sentences (under 30 words); " + TONE[v["genre"]] + "; react to what was just said and to the "
            "moment when it fits, without reciting the card; if asked something you do not know, say so in character; simple English; no emojis, "
            "asterisks or stage directions; never mention being an AI, a character or a game. The player writes casually.\n\n"
            'Return only JSON: {"talks": [{"turns": [{"player": str, "villager": str}, ...]}, ...], "neighbour_talk": [{"neighbour": str, "villager": str}, ...]}')


def gen_plan(genre):
    its = GEN_INTENTS[genre]; n = R.choice([3, 4, 4, 5, 5, 6]); seq = ["greet"] if R.random() < 0.5 else []
    pool = [k for k in its if k not in ("greet", "recall", "goodbye")]
    while len(seq) < n - 1:
        k = R.choice(pool)
        if k in seq: continue
        seq.append(k)
        if k == "self" and R.random() < 0.7: seq.append("recall")
    seq.append("goodbye" if R.random() < 0.4 else R.choice([k for k in pool if k not in seq]))
    return seq[:7]


GBENCH_KEYS = {"friends": ["greet", "doing", "worry", "food", "advice", "neighbour", "gossip", "tease", "feeling", "dream"],
               "dark": ["greet", "who", "place", "rumour", "memory", "help", "gift_other", "feeling", "tease", "despair"]}


def genbench_prompt(v):
    its = GEN_INTENTS[v["genre"]]; ks = GBENCH_KEYS[v["genre"]]
    return (f"Here is a character in {GENRES[v['genre']]}, with the game's state at this moment:\n\n" + gen_text(v, 0) + "\n\n"
            "Write test lines a player might say to this character, one of each, casual and short:\n" + "; ".join(f"{k} ({its[k]})" for k in ks) + ".\n"
            "Return only JSON: {" + ", ".join(f'"{k}": str' for k in ks) + "}")


def vil_card():
    temper = R.choice(list(TEMPERS)); season = R.choice(SEASONS); likes = R.sample(ITEMS, 2); dislike = R.choice([x for x in ITEMS if x not in likes])
    nbs = [(fresh_name(), R.choice(SPECIES), R.choice(list(TEMPERS)), R.choice(RELS)) for _ in range(2)]
    shop = R.choice(SHOPS); o, c = R.randint(6, 10), R.randint(17, 23)
    ev = R.choice(EVENTS).format(nb=nbs[0][0]); mem = R.choice(MEMORY).format(item=R.choice(ITEMS))
    return dict(name=fresh_name(), species=R.choice(SPECIES), temper=temper, catch=R.choice(CATCH_A) + R.choice(CATCH_B), hobby=R.choice(HOBBIES),
                likes=likes, dislike=dislike, nbs=nbs, season=season, weather=R.choice(WEATHERS[season]), time=R.choice(TIMES), event=ev,
                doing=R.choice(DOING), memory=mem, shop=shop, hours=f"{o} am to {c - 12} pm", town=R.choice(["Maple Cove", "Pebble Bay", "Fernhill",
                "Willow Creek", "Sunny Isle", "Acorn Hollow", "Misty Pines", "Clover Field"]))


def vil_text(v, fmt, talking_to=None):
    """the card the game would fill in; two layouts"""
    nb = "; ".join(f"{n} the {s} ({t}, your {r})" for n, s, t, r in v["nbs"])
    lines = [f"You are {v['name']}, a {v['temper']} {v['species']} villager in {v['town']}. You are {TEMPERS[v['temper']]}. "
             f"Your catchphrase is \"{v['catch']}\".",
             f"Hobby: {v['hobby']}. Likes: {v['likes'][0][2:] if v['likes'][0].startswith('a ') else v['likes'][0][3:]}, "
             f"{v['likes'][1].split(' ', 1)[1]}. Dislikes: {v['dislike'].split(' ', 1)[1]}.", f"Neighbours: {nb}.",
             f"Now: {v['season']}, {v['weather']}, {v['time']}. You are {v['doing']}."]
    if v["event"] != "none": lines.append(f"Today: {v['event']}.")
    if v["memory"] != "none": lines.append(v["memory"])
    lines.append(f"{v['shop'].capitalize()} is open {v['hours']}.")
    if talking_to: lines.append(f"You are talking with {talking_to}.")
    return ("\n" if fmt == 0 else " ").join(lines)


def vil_plan():
    n = R.choice([3, 4, 4, 5, 5, 6]); seq = ["greet"] if R.random() < 0.5 else []
    pool = [k for k in VIL_INTENTS if k not in ("greet", "recall", "goodbye")]
    while len(seq) < n - 1:
        k = R.choice(pool)
        if k in seq: continue
        seq.append(k)
        if k == "self" and R.random() < 0.7: seq.append("recall")
    seq.append("goodbye" if R.random() < 0.4 else R.choice([k for k in pool if k not in seq]))
    return seq[:7]


VIL_RULES = ("Rules for the villager's lines: short and lively, 1 or 2 sentences (under 30 words); warm, playful, in the villager's own temper "
             "and voice; say the catchphrase at the end of about one line in three, not every line; react to what was just said and to the "
             "moment (weather, time, today's event, what you are doing, what the player did lately) when it fits, without reciting the card; "
             "react to a gift by what you like or dislike; if asked something you do not know, say so in character; simple everyday English; "
             "no emojis, asterisks or stage directions; never mention being an AI, a character or a game. The player writes casually.")


def vil_prompt(v, plans):
    ds = "\n".join(f"Talk {i + 1} (with the player): the player's lines, in order: " + "; ".join(f"({j + 1}) {VIL_INTENTS[k]}" for j, k in enumerate(p))
                   for i, p in enumerate(plans))
    n0 = v["nbs"][0]
    return ("Here is a villager in a cozy life-sim game, with the game's state at this moment:\n\n" + vil_text(v, 0) + "\n\n"
            f"Write {len(plans)} talks with the player, each a new visit:\n{ds}\n"
            f"Then talk {len(plans) + 1}: {n0[0]} (a {n0[2]} {n0[1]}, {TEMPERS[n0[2]]}) walks up to {v['name']} and they chat about the day, 3 to 5 "
            f"exchanges, {n0[0]} speaking first each time, each in their own voice.\n\n" + VIL_RULES + "\n\n"
            'Return only JSON: {"talks": [{"turns": [{"player": str, "villager": str}, ...]}, ...], '
            '"neighbour_talk": [{"neighbour": str, "villager": str}, ...]}')


def vilbench_prompt(v):
    n0 = v["nbs"][0]
    return ("Here is a villager in a cozy life-sim game, with the game's state at this moment:\n\n" + vil_text(v, 0) + "\n\n"
            "Write test lines a player might say to this villager, one of each, casual and short:\n"
            "greet; weather (about the weather or season); doing (what are you doing?); gift_like (giving one of the things it likes, name it); "
            "gift_dislike (giving the thing it dislikes, name it); advice (asking for a suggestion); "
            f"neighbour (asking about {n0[0]}); event (asking what is going on today); memory (about what the player did lately; if the card has no such "
            "line, about the player's own day); tease (a little rude); feeling (how the player feels).\n"
            'Return only JSON: {"greet": str, "weather": str, "doing": str, "gift_like": str, "gift_dislike": str, "advice": str, "neighbour": str, '
            '"event": str, "memory": str, "tease": str, "feeling": str}')


def vil_rows(v, d):
    out = []
    for dl in (d.get("talks") or []):
        ts = dl.get("turns") if isinstance(dl, dict) else dl
        turns = [(clean(t.get("player", "")), clean(t.get("villager", ""))) for t in (ts or []) if isinstance(t, dict)]
        if len(turns) < 2 or not all(p and ok_reply(r) for p, r in turns): continue
        msgs = [{"role": "system", "content": card_of(v, R.randrange(2))}]
        for p, r in turns: msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        out.append({"messages": msgs, "src": "villager_" + v.get("genre", "cozy")})
    nt = d.get("neighbour_talk")
    nt = nt.get("turns") if isinstance(nt, dict) else nt
    turns = [(clean(t.get("neighbour", "")), clean(t.get("villager", ""))) for t in (nt or []) if isinstance(t, dict)]
    if len(turns) >= 2 and all(p and ok_reply(r) for p, r in turns):
        n0 = v["nbs"][0]; who = f"{n0[0]} the {n0[1]}, your {n0[3]}" if n0[1] else n0[0]
        msgs = [{"role": "system", "content": card_of(v, R.randrange(2), who)}]
        for p, r in turns: msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        out.append({"messages": msgs, "src": "villager_nb_" + v.get("genre", "cozy")})
    return out


def card_of(v, fmt, talking_to=None): return gen_text(v, fmt, talking_to) if v.get("genre", "cozy") != "cozy" else vil_text(v, fmt, talking_to)


def vilbench_row(i, d, v):
    if v.get("genre", "cozy") != "cozy":
        ks = GBENCH_KEYS[v["genre"]]
        if not all(isinstance(d.get(k), str) and d[k].strip() for k in ks): return None
        nm = NAMES_BENCH[i % len(NAMES_BENCH)]
        probes = [{"kind": v["genre"] + ":" + k, "turns": [clean(d[k])]} for k in ks]
        probes.append({"kind": v["genre"] + ":recall", "turns": [f"Hi, I'm {nm}.", "Greetings.", "Do you remember my name?"], "answer": nm})
        return {"card": card_of(v, i % 2), "card_fields": v, "probes": probes, "genre": v["genre"]}
    keys = ["greet", "weather", "doing", "gift_like", "gift_dislike", "advice", "neighbour", "event", "memory", "tease", "feeling"]
    if not all(isinstance(d.get(k), str) and d[k].strip() for k in keys): return None
    probes = [{"kind": k, "turns": [clean(d[k])]} for k in keys]
    nm = NAMES_BENCH[i % len(NAMES_BENCH)]
    probes.append({"kind": "recall", "turns": [f"Hi, I'm {nm}!", "Oh, hi! Nice to meet you!", "Do you remember my name?"], "answer": nm})
    probes.append({"kind": "town", "turns": [f"When is {v['shop']} open?"], "answer": v["hours"].split(" to ")[0]})
    return {"card": vil_text(v, i % 2), "card_fields": v, "probes": probes, "genre": "cozy"}


VJUDGE = ("You grade one line written by a small model playing a character in {genre}.\n\nCHARACTER CARD AND GAME STATE:\n{card}\n\nCONVERSATION SO FAR:\n{conv}\n\nCHARACTER'S LINE TO GRADE:\n{reply}\n\n{note}"
          "Score each from 0 to 2:\n"
          "- answers: responds to what the player just said (2 = directly and sensibly, 1 = partly or vaguely, 0 = ignores it, repeats it or is nonsense)\n"
          "- consistent: agrees with the card and the game state (2 = yes, 1 = small slip, 0 = contradicts it, e.g. loves a gift it dislikes, wrong "
          "weather, wrong shop hours, confuses who is who)\n"
          "- in_character: sounds like THIS character and the game's tone (0 = talks like an AI assistant, mentions being an AI or a game, another persona, or the wrong tone, e.g. cute in a grim game)\n"
          "- fluent: grammatical, coherent English (0 = garbled)\n"
          "- charm: a line a player would enjoy in this game, lively and with flavour (2), plain but fine (1), flat, odd or off-putting (0)\n"
          "Then 'usable': true only if the game could show this exact line at this moment without embarrassment.\n"
          'Return only JSON: {{"answers": int, "consistent": int, "in_character": int, "fluent": int, "charm": int, "usable": bool}}')


FJUDGE = ("You check training data for a small model that plays characters in video games. Below is a character card with the game's "
          "state, and a talk where someone speaks to the character.\n\nCARD:\n{card}\n\nTALK:\n{talk}\n\n"
          "Grade every CHARACTER line (C1, C2, ...) from 0 to 2: 2 = natural, in character, answers what was said, agrees with the card; "
          "1 = fine but plain or a little off; 0 = bad: nonsense, does not answer, wrong speaker (e.g. calls itself by its own name as if "
          "someone else), contradicts the card or the talk, breaks character, or broken English.\n"
          'Return only JSON: {{"grades": [int, ...]}} with one grade per character line, in order.')
FSTRICT = ("Be strict, like an editor choosing lines for a shipped game: give 2 only to a line a player would really enjoy here - natural, "
           "clearly in this character's own voice and tone, answering exactly what was said, true to the card and the moment. Plain, generic, "
           "slightly odd or slightly off-topic lines get 1. Most lines in a typical batch should NOT get 2.\n")


def vfilter(llm):
    rows = [json.loads(l) for l in open(A.inp)]; prompts = []
    for r in rows:
        ms = r["messages"]; card = next((m["content"] for m in ms if m["role"] == "system"), "(none)"); k = 0; talk = []
        for m in ms:
            if m["role"] == "user": talk.append(f"OTHER: {m['content']}")
            elif m["role"] == "assistant": k += 1; talk.append(f"C{k}: {m['content']}")
        q = FJUDGE.format(card=card, talk="\n".join(talk))
        if A.strict: q = q.replace("Return only JSON", FSTRICT + "Return only JSON")
        prompts.append(q)
    texts = run(llm, prompts, 0.0, 120); kept = cut = dropped = unparsed = 0; hist = collections.Counter()
    gf = open(A.out + ".grades.jsonl", "w")   # every talk with its grades: other thresholds later without the judge
    with open(A.out, "w") as f:
        for r, t in zip(rows, texts):
            g = (parse_json(t) or {}).get("grades") if isinstance(parse_json(t), dict) else None
            gf.write(json.dumps({"messages": r["messages"], "src": r.get("src"), "grades": g}) + "\n")
            n_ast = sum(m["role"] == "assistant" for m in r["messages"])
            if not isinstance(g, list) or len(g) != n_ast or not all(isinstance(x, int) for x in g): unparsed += 1; continue
            hist.update(g); bad = next((i for i, x in enumerate(g) if x < A.min_grade), None)
            if bad is None: f.write(json.dumps(r) + "\n"); kept += 1; continue
            if bad < 2: dropped += 1; continue
            out, k = [], 0
            for m in r["messages"]:
                if m["role"] == "assistant":
                    if k == bad: break
                    k += 1
                out.append(m)
            while out and out[-1]["role"] != "assistant": out.pop()
            r = dict(r); r["messages"] = out; f.write(json.dumps(r) + "\n"); cut += 1
    gf.close()
    print(f"[filter] {len(rows)} talks: {kept} kept whole, {cut} cut before a bad line, {dropped} dropped, {unparsed} unparsed; "
          f"line grades {dict(sorted(hist.items()))} -> {A.out}", flush=True)


LJ_HEAD = ("You grade one line spoken by a character in a video game (training data for a small model). Below is the character's card "
           "with the game's state, what was said so far (OTHER = the one talking to the character), and the character's next line.\n\n")
LJ_CHECK = ("Check these first; any one of them makes the line 0:\n"
            "- it is only a name or a word or two, or repeats the other's line or its own earlier line\n"
            "- it addresses someone by the character's OWN name, or mixes up who is who\n"
            "- it ignores what OTHER just said, or answers something else\n"
            "- it contradicts the card: loves a gift the card says it dislikes, hates one it likes, wrong hours, wrong weather or event\n"
            "- it breaks character (mentions being a character, an AI or a game) or has broken English\n"
            "Otherwise: 2 = a line a player would really enjoy here (natural, in this character's own voice and the game's tone, answers well); "
            "1 = fine but plain, generic or a little off.\n")
LJ_TAIL = 'Return only JSON: {{"grade": int}}'


def lj_item(x, with_grade=False):
    talk = "\n".join(("OTHER: " if (len(x["talk"]) - i) % 2 == 1 else "CHARACTER: ") + t for i, t in enumerate(x["talk"])) or "(nothing yet)"
    s_ = f"CARD:\n{x['card']}\n\nSO FAR:\n{talk}\n\nCHARACTER'S LINE:\n{x['line']}\n"
    return s_ + (f"GRADE: {x['grade']}\n" if with_grade else "")


def linejudge(llm):
    rows = [json.loads(l) for l in open(A.inp)]; shots = [json.loads(l) for l in open(A.shots)] if A.shots else []
    sid = {r["id"] for r in shots}; rows = [r for r in rows if r["id"] not in sid]
    ex = ""
    if A.variant == "fewshot" and shots:
        ex = "Examples graded by the game's editor:\n\n" + "\n---\n".join(lj_item(x, True) for x in shots) + "\n=== Now grade this one ===\n\n"
    rule = LJ_CHECK if A.variant in ("checklist", "fewshot") else "2 = good, 1 = fine but plain, 0 = bad.\n"
    prompts = [LJ_HEAD + ex + lj_item(r) + "\n" + rule + LJ_TAIL.format() for r in rows]
    texts = run(llm, prompts, 0.0, 40); n_bad = 0
    with open(A.out, "w") as f:
        for r, t in zip(rows, texts):
            g = (parse_json(t) or {}).get("grade") if isinstance(parse_json(t), dict) else None
            if g not in (0, 1, 2): n_bad += 1; g = 1
            f.write(json.dumps({"id": r["id"], "judge": g, "src": r.get("src")}) + "\n")
    print(f"[linejudge] {A.variant}: {len(rows)} lines, {n_bad} unparsed -> {A.out}", flush=True)


def parse_json(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S); a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a: return None
    try: return json.loads(t[a:b + 1])
    except Exception: return None


def engine():
    from vllm import LLM
    import torch
    # an NVFP4 build wants Blackwell: on older GPUs its fallback goes first; any other model is tried first as asked
    order = (A.fallback, A.model) if "NVFP4" in A.model and torch.cuda.get_device_capability()[0] < 10 else (A.model, A.fallback)
    for m in order:
        kw = dict(model=m, max_model_len=A.max_len, gpu_memory_utilization=A.gpu_mem, enable_prefix_caching=True, seed=A.seed, max_num_seqs=A.max_seqs,
                  enforce_eager=bool(A.eager))
        try:   # text only: the Qwen3.5 models carry a vision encoder that would take memory a 12 GB card does not have
            try: return LLM(language_model_only=True, **kw), m
            except TypeError: return LLM(**kw), m
        except Exception as e:
            print(f"[gen] {m} did not load: {str(e)[:300]}", flush=True)
            if m == order[-1]: raise


def run(llm, prompts, temp, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tokens); out = []; t0 = time.time(); k = A.chunk or len(prompts)
    for a in range(0, len(prompts), k):
        msgs = [[{"role": "user", "content": p}] for p in prompts[a:a + k]]
        try: outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        except TypeError: outs = llm.chat(msgs, sp, use_tqdm=False)
        out += [o.outputs[0].text for o in outs]
        if A.chunk: print(f"[gen] {len(out)}/{len(prompts)} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    return out


def ok_card(c):
    return (isinstance(c, dict) and all(isinstance(c.get(k), str) and c.get(k).strip() for k in ("name", "role", "place", "personality"))
            and isinstance(c.get("facts"), list) and 3 <= len(c["facts"]) <= 9 and all(isinstance(f, str) and 5 < len(f) < 200 for f in c["facts"]))


def ok_reply(s): return 2 <= len(s) <= (440 if A.mode == "help" else 320) and not AI_TALK.search(s) and "*" not in s and sum(ord(ch) > 127 for ch in s) <= 2


def turn_list(dl):
    """a dialogue as the model wrote it: {"turns": [...]} or the list of turns itself"""
    ts = dl.get("turns") if isinstance(dl, dict) else dl if isinstance(dl, list) else None
    return [(clean(t.get("player", "")), clean(t.get("npc", ""))) for t in (ts or []) if isinstance(t, dict)]


def train_rows(i, d, c, plans):
    out = []
    for j, dl in enumerate(d.get("dialogues") or []):
        turns = turn_list(dl)
        if len(turns) < 2 or not all(p and ok_reply(r) for p, r in turns): continue
        msgs = [{"role": "system", "content": card_text(c, R.randrange(3))}]
        for p, r in turns: msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        out.append({"messages": msgs, "card": c, "plan": plans[i][j] if plans[i] and j < len(plans[i]) else []})
    return out


def bench_row(i, d, c, seen):
    fq = [q for q in d.get("fact_questions") or [] if isinstance(q, dict) and q.get("q") and q.get("answer")]
    fu = d.get("followup") if isinstance(d.get("followup"), dict) else {}
    if len(fq) < 2 or c["name"].lower() in seen or not all(isinstance(d.get(k), str) and d[k].strip() for k in ("unknown", "meta", "rude")): return None
    seen.add(c["name"].lower()); probes = []
    for q in fq[:3]: probes.append({"kind": "fact", "turns": [clean(q["q"])], "answer": clean(q["answer"])})
    if all(fu.get(k) for k in ("q", "reference", "q2", "answer")):
        probes.append({"kind": "followup", "turns": [clean(fu["q"]), clean(fu["reference"]), clean(fu["q2"])], "answer": clean(fu["answer"])})
    for k in ("unknown", "meta", "rude"): probes.append({"kind": k, "turns": [clean(d[k])]})
    nm = NAMES_BENCH[i % len(NAMES_BENCH)]
    probes.append({"kind": "recall", "turns": [f"Hi, I'm {nm}. Nice to meet you.", "Nice to meet you too.", "Do you remember my name?"], "answer": nm})
    return {"card": card_text(c, i % 3), "card_fields": c, "probes": probes}


def helpbench_row(i, d, c, seen):
    lst = lambda k: [clean(x) for x in (d.get(k) or []) if isinstance(x, str) and x.strip() and (k in ("smalltalk", "feeling") or x.strip().endswith("?"))]
    dq = [q for q in d.get("directions") or [] if isinstance(q, dict) and q.get("q") and q.get("answer")]
    if c["name"].lower() in seen or not dq or not lst("howto") or not lst("smalltalk") or not lst("feeling"): return None
    seen.add(c["name"].lower()); probes = [{"kind": "directions", "turns": [clean(q["q"])], "answer": clean(q["answer"])} for q in dq[:2]]
    for k in ("howto", "explain", "smalltalk", "feeling"): probes += [{"kind": k, "turns": [x]} for x in lst(k)[:2]]
    if isinstance(d.get("recommend"), str) and d["recommend"].strip().endswith("?"): probes.append({"kind": "recommend", "turns": [clean(d["recommend"])]})
    return {"card": card_text(c, i % 3), "card_fields": c, "probes": probes}


def main():
    T0 = time.time()
    if A.mode == "linejudge":
        llm, used = engine(); print(f"[linejudge] with {used}", flush=True); linejudge(llm); return
    if A.mode == "vfilter":
        llm, used = engine(); print(f"[filter] with {used}", flush=True); vfilter(llm); return
    if A.mode in ("villager", "villagerbench"):
        llm, used = engine(); print(f"[gen] {A.mode} with {used}: {A.n} requests", flush=True)
        gl = A.genres.split(","); vs = [vil_card() if (g := gl[i % len(gl)]) == "cozy" else gen_card(g) for i in range(A.n)]
        seen = set(); vs = [v for v in vs if not (v["name"] in seen or seen.add(v["name"]))]
        if A.mode == "villager":
            plans = [[vil_plan(), vil_plan()] if v.get("genre", "cozy") == "cozy" else [gen_plan(v["genre"]), gen_plan(v["genre"])] for v in vs]
            prompts = [vil_prompt(v, p) if v.get("genre", "cozy") == "cozy" else gen_prompt(v, p) for v, p in zip(vs, plans)]; mt = 2200
        else: plans = [None] * len(vs); prompts = [vilbench_prompt(v) if v.get("genre", "cozy") == "cozy" else genbench_prompt(v) for v in vs]; mt = 700
        n_ok = n_bad = 0; k = A.chunk or len(prompts); fo = open(A.out, "w"); fr = open(A.out + ".raw", "w")
        for a in range(0, len(prompts), k):   # chunk by chunk, each written at once: a box that dies keeps what it made
            texts = run(llm, prompts[a:a + k], 0.9 if A.mode == "villager" else 0.8, mt)
            for v, p_, t in zip(vs[a:a + k], plans[a:a + k], texts):
                fr.write(json.dumps({"spec": v, "plans": p_, "text": t}) + "\n")
                d = parse_json(t)
                if not isinstance(d, dict): n_bad += 1; continue
                try: rows = vil_rows(v, d) if A.mode == "villager" else [x for x in [vilbench_row(a, d, v)] if x]
                except Exception as e: print(f"[gen] skipped: {type(e).__name__}: {str(e)[:100]}", flush=True); n_bad += 1; continue
                for x in rows: fo.write(json.dumps(x) + "\n")
                n_ok += len(rows); n_bad += not rows
            fo.flush(); fr.flush(); print(f"[gen] {A.mode}: {n_ok} written, {n_bad} dropped after {a + len(texts)} requests [{(time.time() - T0) / 60:.1f} min]", flush=True)
        fo.close(); fr.close(); return
    if A.mode in ("train", "bench", "help", "helpbench"):
        if A.from_raw:
            raw = [json.loads(l) for l in open(A.from_raw)]; specs = [r["spec"] for r in raw]; plans = [r.get("plans") for r in raw]; texts = [r["text"] for r in raw]
            print(f"[gen] {A.mode}: {len(raw)} outputs from {A.from_raw}", flush=True)
        else:
            llm, used = engine(); print(f"[gen] {A.mode} with {used}: {A.n} requests", flush=True)
            specs = [card_spec() for _ in range(A.n)]
            if A.mode == "train":
                plans = [[plan(), plan()] for _ in specs]; texts = run(llm, [train_prompt(s, p) for s, p in zip(specs, plans)], 0.9, 2600)
            elif A.mode == "help":
                plans = [[help_plan(), help_plan()] for _ in specs]; texts = run(llm, [help_prompt(s, p) for s, p in zip(specs, plans)], 0.9, 3000)
            elif A.mode == "helpbench":
                plans = [None] * len(specs); texts = run(llm, [helpbench_prompt(s) for s in specs], 0.8, 1600)
            else:
                plans = [None] * len(specs); texts = run(llm, [bench_prompt(s) for s in specs], 0.8, 1600)
            with open(A.out + ".raw", "w") as f:   # what the model wrote, kept before any parsing
                for s_, p_, t_ in zip(specs, plans, texts): f.write(json.dumps({"spec": s_, "plans": p_, "text": t_}) + "\n")
        n_ok = n_bad = 0; seen = set()
        with open(A.out, "w") as f:
            for i, t in enumerate(texts):
                try:
                    d = parse_json(t)
                    if isinstance(d, dict) and not ok_card(d): d = next((v for v in d.values() if isinstance(v, dict) and ok_card(v)), d)   # wrapped once
                    if not ok_card(d): n_bad += 1; continue
                    c = {k: clean(d[k]) for k in ("name", "role", "place", "personality")}; c["facts"] = [clean(x) for x in d["facts"]]
                    if A.mode in ("train", "help"): rows = train_rows(i, d, c, plans)
                    else: rows = [x for x in [(bench_row if A.mode == "bench" else helpbench_row)(i, d, c, seen)] if x]
                except Exception as e:
                    print(f"[gen] output {i} skipped: {type(e).__name__}: {str(e)[:100]}", flush=True); n_bad += 1; continue
                for x in rows: f.write(json.dumps(x) + "\n")
                n_ok += len(rows); n_bad += (2 if A.mode in ("train", "help") else 1) - len(rows)
        print(f"[gen] {A.mode}: {n_ok} written, {n_bad} dropped [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)
    else:
        rows = [json.loads(l) for l in open(A.inp)]
        llm, used = engine(); prompts = []
        for r in rows:
            conv = "\n".join(("PLAYER: " if i % 2 == 0 else "NPC: ") + t for i, t in enumerate(r["turns"]))
            note = f"(For reference, the correct reply should contain: {r['answer']}.)\n\n" if r.get("answer") else ""
            g_ = r.get("genre") or ("cozy" if "catchphrase" in r["card"] else None)
            if g_: prompts.append(VJUDGE.format(genre=GENRES[g_], card=r["card"], conv=conv, reply=r["reply"] or "(empty)", note=note))
            else: prompts.append(JUDGE.format(card=r["card"], conv=conv, reply=r["reply"] or "(empty)", note=note))
        texts = run(llm, prompts, 0.0, 120); n_bad = 0
        with open(A.out, "w") as f:
            for r, t in zip(rows, texts):
                g = parse_json(t) or {}
                if not isinstance(g, dict) or "usable" not in g: n_bad += 1; g = g if isinstance(g, dict) else {}
                r["judge"] = {k: g.get(k) for k in ("answers", "consistent", "in_character", "fluent", "charm", "usable")}; f.write(json.dumps(r) + "\n")
        print(f"[judge] {len(rows)} replies graded with {used}, {n_bad} unparsed [{(time.time() - T0) / 60:.1f} min] -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
