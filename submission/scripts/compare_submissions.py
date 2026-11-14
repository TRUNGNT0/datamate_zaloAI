"""So hai lần chạy cùng một đầu vào: tỷ lệ câu có đáp án giống hệt (kể cả cùng abstain). Chỉ dùng thư viện chuẩn."""
import json
import sys


def load(path):
    return {r["id"]: (r.get("type"), json.dumps(r.get("answer"), sort_keys=True, ensure_ascii=False)) for r in map(json.loads, open(path, encoding="utf-8")) if r.get("id")}


a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) | set(b))
same = sum(1 for i in ids if a.get(i) == b.get(i))
print(f"{same}/{len(ids)} câu giống hệt ({100 * same / max(1, len(ids)):.2f}%). Khác nhau: {len(ids) - same}")
for i in [i for i in ids if a.get(i) != b.get(i)][:10]:
    print("  khác:", i, a.get(i), "|", b.get(i))
