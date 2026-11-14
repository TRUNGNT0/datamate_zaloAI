"""Tính PA của một lần chạy trên split CÓ answers.jsonl (vd train) bằng đúng bộ chấm của repo.
Chạy trong image:  docker run --rm --entrypoint python3 -v <split>:/data:ro -v <ra>:/result:ro IMAGE /code/tools/score_submission.py"""
import json
import sys

sys.path.insert(0, "/code")
import pandas as pd  # noqa: E402

from evaluation import scoring  # noqa: E402


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data"
    result_dir = sys.argv[2] if len(sys.argv) > 2 else "/result"
    subs = [json.loads(line) for line in open(f"{result_dir}/submission.jsonl", encoding="utf-8") if line.strip()]
    logs = [json.loads(line) for line in open(f"{result_dir}/logs/questions.jsonl", encoding="utf-8") if line.strip()]
    answers = pd.read_json(f"{data_dir}/answers.jsonl", lines=True)
    answers = answers[answers["id"].isin({s["id"] for s in subs})]
    evaluated = scoring.evaluate(subs, logs, answers)
    print(scoring.summary_text(evaluated))


if __name__ == "__main__":
    main()
