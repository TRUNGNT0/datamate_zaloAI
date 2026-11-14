"""Tạo, ghi và kiểm định file nộp submission.jsonl."""
from __future__ import annotations

import json
import os

from datamate.tools.results import VALID_TYPES

MAX_BYTES = 5 * 1024 * 1024


def make_sub(qid, typ, ans) -> dict:
    if typ in (None, "abstain"):
        return {"id": qid, "type": "abstain"}
    if typ == "unanswerable":
        return {"id": qid, "type": typ, "reason": str(ans)[:300]}
    return {"id": qid, "type": typ, "answer": ans}


def dump_line(sub: dict) -> str:
    """Một dòng JSON chuẩn (cấm NaN/Infinity). Không ghi được thì chuyển câu đó thành abstain thay vì hỏng cả file."""
    try:
        return json.dumps(sub, ensure_ascii=False, default=str, allow_nan=False)
    except ValueError:
        return json.dumps({"id": sub["id"], "type": "abstain"}, ensure_ascii=False)


def fill_missing(subs: list[dict], ids) -> list[dict]:
    """Mỗi id trong questions.jsonl có đúng một dòng, theo thứ tự file câu hỏi; câu chưa chạy = abstain."""
    by_id = {s["id"]: s for s in subs}
    return [by_id.get(i) or {"id": i, "type": "abstain"} for i in ids]


def write_submission(subs: list[dict], ids, path: str) -> list[dict]:
    rows = fill_missing(subs, ids)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(dump_line(s) + "\n")
    return rows


def _bad_const(c):
    raise ValueError(f"hằng không hợp lệ trong JSON: {c}")


def validate_submission(path: str, ids) -> list[str]:
    """Kiểm nhanh trước khi nộp: JSON chuẩn, đủ và duy nhất id, type hợp lệ, có answer khi cần, dưới 5MB."""
    problems, seen = [], set()
    if os.path.getsize(path) > MAX_BYTES:
        problems.append("file > 5MB")
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        try:
            d = json.loads(line, parse_constant=_bad_const)
        except ValueError as e:
            problems.append(f"dòng {n}: {e}")
            continue
        if d.get("id") in seen:
            problems.append(f"dòng {n}: id trùng {d.get('id')}")
        seen.add(d.get("id"))
        t = d.get("type")
        if t not in VALID_TYPES | {"abstain"}:
            problems.append(f"dòng {n}: type lạ {t!r}")
        elif t not in ("abstain", "unanswerable") and d.get("answer") is None:
            problems.append(f"dòng {n}: thiếu answer")
    problems += [f"thiếu id {i}" for i in list(set(ids) - seen)[:5]]
    problems += [f"id lạ {i}" for i in list(seen - set(ids))[:5]]
    return problems


def check(sub_path, questions_path) -> list[str]:
    from pathlib import Path
    q_ids = [json.loads(l)["id"] for l in Path(questions_path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return validate_submission(str(sub_path), q_ids)


def merge_shards(out_dir, questions_path, shards=None, quiet=False) -> None:
    from pathlib import Path
    out_dir = Path(out_dir)
    shard_dirs = sorted(shards.glob("shard*")) if shards else sorted(out_dir.glob("shard*"))
    all_subs = {}
    for sd in shard_dirs:
        sub_file = sd / "submission.jsonl"
        if sub_file.exists():
            for line in sub_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("type") != "abstain" or r["id"] not in all_subs:
                        all_subs[r["id"]] = r
    q_ids = [json.loads(l)["id"] for l in Path(questions_path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    write_submission(list(all_subs.values()), q_ids, str(out_dir / "submission.jsonl"))
    if not quiet:
        print(f"Merged {len(shard_dirs)} shards into {out_dir / 'submission.jsonl'}")

