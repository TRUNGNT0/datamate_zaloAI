"""Kiểm đầu ra của một lần chạy theo yêu cầu của ban tổ chức: submission.jsonl đủ id + type hợp lệ, và mỗi câu có bản ghi log
với id, answer, elapsed_ms (số nguyên >= 0), thinking (có khóa). Chỉ dùng thư viện chuẩn. Thoát 1 nếu có lỗi."""
import argparse
import json
import os
import sys

VALID = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions")
    parser.add_argument("result_dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ids = [json.loads(line)["id"] for line in open(args.questions, encoding="utf-8-sig") if line.strip()]
    if args.limit:
        ids = ids[: args.limit]
    problems = []
    sub_path = os.path.join(args.result_dir, "submission.jsonl")
    if not os.path.isfile(sub_path):
        problems.append("thiếu submission.jsonl")
        subs = {}
    else:
        subs = {}
        for n, line in enumerate(open(sub_path, encoding="utf-8"), 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                problems.append(f"submission dòng {n}: {error}")
                continue
            subs[row.get("id")] = row
            if row.get("type") not in VALID:
                problems.append(f"submission dòng {n}: type lạ {row.get('type')!r}")
            elif row["type"] not in ("abstain", "unanswerable") and row.get("answer") is None:
                problems.append(f"submission dòng {n}: thiếu answer")
    problems += [f"submission thiếu id {i}" for i in ids if i not in subs][:5]
    log_path = os.path.join(args.result_dir, "logs", "questions.jsonl")
    logs = {}
    if not os.path.isfile(log_path):
        problems.append("thiếu logs/questions.jsonl")
    else:
        for n, line in enumerate(open(log_path, encoding="utf-8"), 1):
            try:
                record = json.loads(line)
            except ValueError as error:
                problems.append(f"log dòng {n}: {error}")
                continue
            logs[record.get("id")] = record
            ms = record.get("elapsed_ms")
            if not (isinstance(ms, int) and not isinstance(ms, bool) and ms >= 0):
                problems.append(f"log {record.get('id')}: elapsed_ms không hợp lệ ({ms!r})")
            if "thinking" not in record or "answer" not in record:
                problems.append(f"log {record.get('id')}: thiếu thinking/answer")
    problems += [f"log thiếu id {i}" for i in ids if i not in logs][:5]
    answered = sum(1 for row in subs.values() if row.get("type") != "abstain")
    print(f"{len(subs)} dòng nộp ({answered} có đáp án, {len(subs) - answered} abstain) | {len(logs)} bản ghi log")
    if problems:
        print("LỖI:\n  " + "\n  ".join(problems[:20]))
        return 1
    print("ĐẦU RA HỢP LỆ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
