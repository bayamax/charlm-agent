# charlm-agent

5000 万パラメータの**文字単位**の小さな言語モデルです。役になりきった会話と、ゲームの行動選び（目標・状態・操作の一覧を読んで、操作をタグで返す）の 2 つをこなします。

**学習済みモデル：** https://huggingface.co/baya1116/charlm-agent（成績、使い方、ゲームに組み込むときのコツもこちら）

## モデル

`models/` に 4bit 版（各 33MB）が入っています。通常版（210MB）は GitHub に載らないので Hugging Face にあります。

| ファイル | 中身 |
|---|---|
| `models/agent14_4bit.pt` | **最新**。agent13b の返答を採点役が選び、最良だけで追加学習（雑談の魅力 1.52 → 1.62、事実の正答 85 → 88%）。迷ったらこれ。嫌いな贈り物への反応だけは 13b のほうが良い（69% 対 49%） |
| `models/agent13b_4bit.pt` | 1 つ前の版。agent12a に、教師モデルに「口調がどうあれ事実は正確に」と指示して書かせた会話を追加学習（事実の正答 80 → 85%、ダークファンタジーでの名前の記憶 23 → 57%、嫌いな贈り物への反応 54 → 69%）。 |
| `models/agent12a_4bit.pt` | 2 つ前の版。agent11 が自分で作った返答の良いものだけで追加学習（事実の正答 74 → 80%） |
| `models/agent11_4bit.pt` | 3 つ前の版 |
| `models/agent6_4bit.pt` | 4 つ前の版 |
| `models/npc4_4bit.pt` | 会話だけに絞った版 |

## すぐ試す

```bash
pip install torch
python3 village.py --model models/agent14_4bit.pt --days 4 --out village.md   # 3 人の村人が 4 日暮らす（CPU で 1〜2 分）
python3 goal_gen.py play --model models/agent14_4bit.pt --n 100 --split test   # 行動選びの採点（見たことのない名前の問題）
```

```python
from charlm_run import Model
m = Model("models/agent14_4bit.pt")
print(m.reply(["You are Tob, the blacksmith of Elm.\n\nHello! Can you fix my sword?"], temp=0.7))
print(m.reply(["GOAL: life >= 1\nOPS: <a> life+1 gold-3 | <b> gold+2 | <c> life-1\nSTATE: life=0 gold=7"], n=260))
```

## ファイル

| 役割 | ファイル | 内容 |
|---|---|---|
| **使う** | `charlm_run.py` | モデルを読んで返答する（通常版・4bit 版、GPU/CPU）。これと `charlm_fold.py` だけあれば動く |
| | `charlm_fold.py` | 文字の表（100 種）と、Unicode を ASCII に寄せる変換 |
| | `village.py` | 村のシミュレーション。ゲームへの組み込み方の見本 |
| | `charlm_chat.py` | 人物設定と台詞を渡して返答の候補を複数出す（通常版のモデル用） |
| **事前学習** | `charlm_prep.py` | WikiText-103 / FineWeb-Edu / SmolTalk を文字列のデータにする |
| | `charlm.py`, `charlm2.py` | 事前学習（層を 2 → 16 に増やしながら学習） |
| **自己学習** | `rft_prep.py`, `charlm_dpo.py` | モデル自身の返答を採点して、良いものだけで学習（RFT）／良い・悪いの差を教える（DPO） |
| **追加学習** | `charlm_sft.py` | 会話の追加学習。途中保存と再開（`--ckpt-every`）、最後に 4bit 化まで |
| | `charlm_gptq.py`, `q4.py` | 4bit 量子化（GPTQ） |
| **データ作り** | `goal_gen.py` | 行動選びの問題と、考えの筋道つきの答えを作る（`data`）／モデルに解かせて採点（`play`） |
| | `npc_gen.py` | 大きめのモデル（vLLM で Qwen）で NPC 会話・案内データを作る／返答を採点する |
| | `soda_prep.py` | SODA（人物設定つきの会話）を学習用に変換 |
| | `agent_gen.py` | 初期の「発言 + 行動タグ」形式のデータ |
| **品質** | `talk_rules.py` | 採点役なしで壊れた行と事実の間違いを見つける（データの選別、返答の足切り） |
| | `judge_audit.py`, `eval/gold_villager_lines.jsonl` | 採点役の点数を人の目の正解セット（150 行）と照合する |
| **採点** | `chat_bits.py` | 会話の予測誤差（雑談の力が落ちていないかの確認） |
| | `npc_bench.py` | NPC の返答を作って並べる、採点の集計 |

## 学習の流れ（agent6 まで）

1. 事前学習：`charlm_prep.py` → `charlm2.py`（63 億文字、16 層）
2. 会話：`charlm_sft.py` で SmolTalk → NPC データ → SODA（npc4）
3. 行動選び：`goal_gen.py data` の問題を雑談と半々に混ぜて `charlm_sft.py`（agent4 → agent5 → agent6）

細かい経緯（各段階の成績、うまくいかなかったこと）は、実験の元になったリポジトリの `archive/charlm/` にあります。
