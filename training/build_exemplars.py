"""Dá»±ng kho vÃ­ dá»¥ few-shot tá»« lá»i giáº£i Ä‘Ã£ Ä‘Ãºng trÃªn táº­p train (chá»‰ dÃ¹ng train, khÃ´ng dÃ¹ng test)."""
from __future__ import annotations

import pandas as pd

from datamate.agent.exemplars import ExemplarBank
from datamate.ingestion.datasets.loader import Split
from training.export_sft import correct_solutions


def build_exemplar_bank(ev: pd.DataFrame, logs: list[dict], split: Split, path: str) -> int:
    """Má»—i dÃ²ng: {"id", "business_id", "question", "type", "code"}. Tráº£ sá»‘ vÃ­ dá»¥ Ä‘Ã£ ghi."""
    if split.answers is None:
        raise ValueError("chá»‰ Ä‘Æ°á»£c dá»±ng kho vÃ­ dá»¥ tá»« táº­p cÃ³ Ä‘Ã¡p Ã¡n chuáº©n (train)")
    qrow = split.questions.set_index("id")
    items = [{"id": r.id, "business_id": qrow.loc[r.id].business_id, "question": qrow.loc[r.id].question,
              "type": r.type, "code": code.strip()} for r, code in correct_solutions(ev, logs)]
    ExemplarBank.dump(items, path)
    return len(items)

