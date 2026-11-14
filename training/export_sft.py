"""Xuáº¥t dá»¯ liá»‡u SFT báº±ng rejection sampling: giá»¯ lá»i giáº£i cá»§a cÃ¢u TRAIN Ä‘Ã£ cho Ä‘Ãºng Ä‘Ã¡p Ã¡n chuáº©n."""
from __future__ import annotations

import json

import pandas as pd

from datamate.agent.context import ContextBuilder
from datamate.agent.prompts import build_system
from datamate.ingestion.datasets.loader import Split
from datamate.tools.results import canon
from evaluation.scoring import match

# Máº«u sinh sau má»™t láº§n lá»—i thÆ°á»ng chá»©a bÃ¬nh luáº­n vá» lá»—i cÅ© mÃ  prompt SFT khÃ´ng cÃ³: dáº¡y model há»c bÃ¬nh luáº­n vÃ´ nghÄ©a
BAD_MARKERS = ("KeyError", "Sá»­a lá»—i", "Láº§n trÆ°á»›c", "Traceback", "lá»—i:")
MAX_COMMENT_LINES = 6
MAX_CODE_CHARS = 3000


def is_clean_code(code: str) -> bool:
    """Code Ä‘á»§ sáº¡ch Ä‘á»ƒ há»c: khÃ´ng bÃ¬nh luáº­n vá» lá»—i, khÃ´ng quÃ¡ nhiá»u bÃ¬nh luáº­n, khÃ´ng quÃ¡ dÃ i."""
    if any(marker in code for marker in BAD_MARKERS) or len(code) > MAX_CODE_CHARS:
        return False
    return sum(1 for line in code.splitlines() if line.strip().startswith("#")) <= MAX_COMMENT_LINES


def correct_solutions(ev: pd.DataFrame, logs: list[dict]):
    """Sinh (dÃ²ng Ä‘Ã¡nh giÃ¡, code) cho má»—i cÃ¢u Ä‘Ã£ tráº£ ÄÃšNG báº±ng LLM; code láº¥y tá»« máº«u cho Ä‘Ãºng káº¿t quáº£ Ä‘Ã£ Ä‘Æ°á»£c chá»n."""
    by_id = {lg["id"]: lg for lg in logs}
    for r in ev[ev.status == "correct"].itertuples():
        lg = by_id[r.id]
        if lg["source"] != "llm":
            continue
        want = canon(r.type, r.answer)
        alive = [x for x in lg["sample_results"] if x is not None]     # `codes` chá»‰ chá»©a máº«u khÃ´ng-None
        codes = [c for c in dict.fromkeys(c for c, x in zip(lg["codes"], alive) if x == want) if is_clean_code(c)]
        if codes:
            yield r, codes[0]


def correct_samples(ev: pd.DataFrame, logs: list[dict], per_question: int = 2, first_attempt_only: bool = True):
    """Sinh (dÃ²ng Ä‘Ã¡nh giÃ¡, code) cho Má»ŒI máº«u cho Ä‘Ãºng Ä‘Ã¡p Ã¡n chuáº©n (khÃ´ng cáº§n Ä‘á»“ng thuáº­n), tá»‘i Ä‘a per_question máº«u
    má»—i cÃ¢u. Cáº§n log do cháº¡y vá»›i collect_all (má»—i máº«u cÃ³ `val`). Chá»‰ láº¥y máº«u Ä‘Ãºng ngay láº§n sinh Ä‘áº§u (khÃ´ng qua tá»± sá»­a)."""
    by_id = {lg["id"]: lg for lg in logs}
    for r in ev.itertuples():
        picked: list[str] = []
        alias = r.alias if isinstance(r.alias, list) else None
        for s in by_id[r.id].get("samples") or []:
            if not s or (first_attempt_only and s.get("attempt", 0) != 0) or not is_clean_code(s["code"]):
                continue
            if s["code"] in picked or not match(s["val"], r.gold, alias, s["type"], r.gold_type):
                continue
            picked.append(s["code"])
            if len(picked) >= per_question:
                break
        for code in picked:
            yield r, code


def export_sft(ev: pd.DataFrame, logs: list[dict], split: Split, builder: ContextBuilder, path: str,
               all_samples: bool = False, per_question: int = 2) -> int:
    """Má»—i dÃ²ng: {"messages": [system, user(ngá»¯ cáº£nh), assistant(code)]}. Chá»‰ dÃ¹ng táº­p train, KHÃ”NG dÃ¹ng test.
    all_samples=True: láº¥y má»i máº«u Ä‘Ãºng (cáº§n cháº¡y collect_all); máº·c Ä‘á»‹nh chá»‰ lá»i giáº£i cá»§a quyáº¿t Ä‘á»‹nh Ä‘á»“ng thuáº­n."""
    if split.answers is None:
        raise ValueError("chá»‰ Ä‘Æ°á»£c dá»±ng dá»¯ liá»‡u SFT tá»« táº­p cÃ³ Ä‘Ã¡p Ã¡n chuáº©n (train)")
    qrow = split.questions.set_index("id")
    system = build_system(builder.cfg)
    source = correct_samples(ev, logs, per_question) if all_samples else correct_solutions(ev, logs)
    n_rows = 0
    with open(path, "w", encoding="utf-8") as f:
        for r, code in source:
            q = qrow.loc[r.id]
            ctx, _ = builder.build(q.question, q.as_of_date, split.business(q.business_id), split, q.business_id)
            f.write(json.dumps({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": ctx},
                {"role": "assistant", "content": "```python\n" + code.strip() + "\n```"}]},
                ensure_ascii=False) + "\n")
            n_rows += 1
    return n_rows

