#!/usr/bin/env python3
"""Preference tuning (DPO) of the char model on its own replies: for a situation (the card, the talk so far) a reply the
checks liked (chosen) and one they did not (rejected); the policy learns to raise the first against the second relative
to a frozen reference, with a little plain learning on the chosen replies and on anchor conversations (chat, goal
episodes) so the rest of what it can do stays.

  python3 charlm_dpo.py --base agent12a/model_final.pt --pairs pairs.jsonl --anchor anchor.jsonl --out agent12 --beta 0.1
  then charlm_sft.py --base agent12/model_final.pt --epochs 0 ... makes the 4-bit file
"""
import argparse, json, math, os, random, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--ref", default="", help="the frozen reference (default: the base)")
ap.add_argument("--pairs", required=True, help='{"card", "turns", "chosen", "rejected"} per line')
ap.add_argument("--anchor", default="", help='{"messages": [...]} conversations learned plainly beside the pairs')
ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--anchor-batch", type=int, default=4)
ap.add_argument("--micro", type=int, default=2, help="pairs per forward pass (the batch is accumulated over them; halved on an out-of-memory)")
ap.add_argument("--sft-w", type=float, default=0.2, help="weight of plain learning on the chosen replies"); ap.add_argument("--anchor-w", type=float, default=0.5)
ap.add_argument("--max-ctx", type=int, default=1600, help="characters of context kept (the end of it)"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val", default="", help='held-out {"messages"} conversations: bits per assistant character every --val-every steps, the best kept as model_best.pt')
ap.add_argument("--val-every", type=int, default=40); ap.add_argument("--val-n", type=int, default=200)
A = ap.parse_args()
random.seed(A.seed); torch.manual_seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charlm_fold as FM  # noqa: E402
CHARS = FM.CHARS; UNK = len(CHARS); EOD = UNK + 1; USR = EOD + 1; AST = USR + 1
os.makedirs(A.out, exist_ok=True); LOG = open(os.path.join(A.out, "train.log"), "a")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

CK = torch.load(A.base, map_location="cpu", weights_only=False); BA = CK["args"]; D, HEADS, SEQ = BA["d"], BA["heads"], BA["seq"]; FF = BA.get("ff") or 4 * D


class Block(nn.Module):   # the same layout and names as charlm_sft.py's
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, FF); s.fc2 = nn.Linear(FF, D)
        s.qk = bool(BA.get("qknorm", 0))
        if s.qk: s.qn = nn.LayerNorm(D // HEADS); s.kn = nn.LayerNorm(D // HEADS)

    def forward(s, x):
        B, T, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sh = lambda z: z.view(B, T, HEADS, D // HEADS)
        q, k, v = sh(q), sh(k), sh(v)
        if s.qk: q, k = s.qn(q), s.kn(k)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a)
        return x + s.fc2(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))


class CharLM(nn.Module):
    def __init__(s, layers, vocab):
        super().__init__()
        s.emb = nn.Embedding(vocab, D); s.pos = nn.Embedding(SEQ, D)
        s.blocks = nn.ModuleList([Block() for _ in range(layers)]); s.ln_f = nn.LayerNorm(D); s.head = nn.Linear(D, vocab)

    def forward(s, ids):
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in s.blocks: x = b(x)
        return s.head(s.ln_f(x))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False); m = CharLM(ck["layers"], ck["model"]["emb.weight"].shape[0]); m.load_state_dict(ck["model"]); return m.to(DEV)


pol = load(A.base); ref = load(A.ref or A.base); ref.eval()
for p in ref.parameters(): p.requires_grad_(False)
enc = lambda t: FM.encode(t.strip()).tolist()


def ctx_ids(card, turns):
    ids = []
    for i, t in enumerate(turns): ids += [USR if i % 2 == 0 else AST] + enc((card + "\n\n" + t) if i == 0 and card else t)
    ids = ids + [AST]
    return ids[-A.max_ctx:]


def seq(ctx, reply):
    """ids and the mask of the reply's characters and the turn's end after it"""
    r = enc(reply) + [USR]; ids = (ctx + r)[-SEQ:]; m = ([0] * len(ctx) + [1] * len(r))[-SEQ:]
    return ids, m


def batch_logp(model, seqs):
    T = max(len(s[0]) for s in seqs); ids = torch.full((len(seqs), T), EOD, dtype=torch.long); mk = torch.zeros(len(seqs), T)
    for i, (a, m) in enumerate(seqs): ids[i, :len(a)] = torch.tensor(a); mk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    ids, mk = ids.to(DEV), mk.to(DEV)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        lg = model(ids[:, :-1])
    lp = -F.cross_entropy(lg.float().transpose(1, 2), ids[:, 1:], reduction="none") * mk[:, 1:]
    return lp.sum(1), mk[:, 1:].sum(1)


def conv_seq(msgs):
    """an anchor conversation: every assistant turn learned (as charlm_sft.py does), cut to the window"""
    ids, m, sys_ = [], [], ""
    for x in msgs:
        if x["role"] == "system": sys_ += x["content"].strip() + "\n\n"; continue
        c = x["content"]
        if x["role"] == "user" and sys_: c = sys_ + c; sys_ = ""
        a = enc(c); ids += [USR if x["role"] == "user" else AST] + a; m += [0] + [x["role"] == "assistant"] * len(a)
    return ids[:min(SEQ, 2048)], m[:min(SEQ, 2048)]


VAL = [conv_seq(json.loads(l)["messages"]) for l in list(open(A.val))[:A.val_n]] if A.val else []


@torch.no_grad()
def val_bits():
    pol.eval(); tot = n = 0.0
    for i in range(0, len(VAL), 4):
        lp, cnt = batch_logp(pol, VAL[i:i + 4]); tot += float(-lp.sum()); n += float(cnt.sum())
    pol.train(); return tot / max(n, 1) / math.log(2)


def save(path):
    sd = {k: v.detach().cpu() for k, v in pol.state_dict().items()}; ck = dict(CK); ck["model"] = sd; torch.save(ck, path)


pairs = [json.loads(l) for l in open(A.pairs)]; random.shuffle(pairs)
anchor = [json.loads(l)["messages"] for l in open(A.anchor)] if A.anchor else []
P = [(ctx_ids(p["card"], p["turns"]), p["chosen"], p["rejected"]) for p in pairs]
steps = int(A.epochs * len(P) / A.batch); opt = torch.optim.AdamW(pol.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0)
out(f"[dpo] {len(P)} pairs, {len(anchor)} anchor conversations; {steps} steps of {A.batch} pairs + {A.anchor_batch if anchor else 0} anchors ({A.micro} per forward); beta {A.beta}, lr {A.lr}")
T0 = time.time(); acc = []; ptr = 0; MICRO = A.micro
best = val_bits() if VAL else None
if VAL: out(f"[dpo] held-out before: {best:.3f} bits per assistant character ({len(VAL)} conversations)")
for step in range(steps):
    lr = A.lr * min(1.0, (step + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, steps))))
    for g in opt.param_groups: g["lr"] = lr
    if ptr + A.batch > len(P): random.shuffle(P); ptr = 0
    b = P[ptr:ptr + A.batch]; ptr += A.batch
    opt.zero_grad(set_to_none=True); margins = []
    try:
        for i in range(0, len(b), MICRO):   # the batch in micro-batches, the gradient accumulated
            mb = b[i:i + MICRO]; ch = [seq(c, x) for c, x, _ in mb]; rj = [seq(c, y) for c, _, y in mb]
            with torch.no_grad(): rc, _ = batch_logp(ref, ch); rr, _ = batch_logp(ref, rj)
            pc, nc = batch_logp(pol, ch); pr, _ = batch_logp(pol, rj)
            margin = A.beta * ((pc - rc) - (pr - rr)); loss = (-F.logsigmoid(margin).sum() + A.sft_w * (-(pc / nc).sum())) / len(b)
            loss.backward(); margins.append(margin.detach())
        for i in range(0, A.anchor_batch if anchor else 0, MICRO):
            k = min(MICRO, A.anchor_batch - i); al, an = batch_logp(pol, [conv_seq(random.choice(anchor)) for _ in range(k)])
            (A.anchor_w * (-(al.sum() / an.sum())) * k / A.anchor_batch).backward()
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); MICRO = max(1, MICRO // 2); out(f"[oom] micro -> {MICRO}"); ptr -= A.batch; continue
    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); margin = torch.cat(margins)
    acc.append((float((margin > 0).float().mean()), float(margin.mean())))
    if (step + 1) % 25 == 0 or step + 1 == steps:
        a_, m_ = zip(*acc); out(f"[dpo] step {step + 1}/{steps}: chosen above rejected {100 * sum(a_) / len(a_):.0f}%, margin {sum(m_) / len(m_):.2f}, lr {lr:.1e}, {(time.time() - T0) / 60:.1f} min"); acc = []
    if VAL and ((step + 1) % A.val_every == 0 or step + 1 == steps):
        v = val_bits(); out(f"[dpo] held-out at step {step + 1}: {v:.3f} bits" + (" (best, saved)" if v < best else ""))
        if v < best: best = v; save(os.path.join(A.out, "model_best.pt"))
save(os.path.join(A.out, "model_final.pt"))
if VAL: out(f"[dpo] held-out best {best:.3f} bits -> model_best.pt")
out(f"[dpo] saved {os.path.join(A.out, 'model_final.pt')} [{(time.time() - T0) / 60:.1f} min]")
