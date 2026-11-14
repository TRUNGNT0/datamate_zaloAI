"""Sinh data_manifest.json: đường dẫn cục bộ, kích thước, SHA-256 của dữ liệu huấn luyện và trọng số, kèm URL/phiên bản công khai
của model gốc (yêu cầu của Docker guideline mục 8). Chỉ dùng thư viện chuẩn."""
import argparse
import hashlib
import json
import os


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root: str, base: str) -> list[dict]:
    files = []
    for folder, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(folder, name)
            files.append({"path": os.path.relpath(path, base), "size": os.path.getsize(path), "sha256": sha256(path)})
    return sorted(files, key=lambda item: item["path"])


def count_lines(path: str) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip()) if os.path.isfile(path) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train = os.path.join(args.data_dir, "train")
    manifest = {
        "base_model": {"name": args.repo, "public_url": f"https://huggingface.co/{args.repo}", "revision": args.revision,
                       "license": "apache-2.0", "parameters": 9653104368, "original_release": "2026-02/03 (trước 26/10/2026)",
                       "inference_assets": "nằm trong image tại /code/models/qwen3.5-9b (không cần tải lại)",
                       "files": walk(args.model_dir, args.model_dir)},
        "training_data": {"source": "dữ liệu train do ban tổ chức cung cấp (dataset/train); không dùng public/private test",
                          "n_questions": count_lines(os.path.join(train, "questions.jsonl")),
                          "n_answers": count_lines(os.path.join(train, "answers.jsonl")),
                          "n_businesses": len(os.listdir(os.path.join(train, "data"))) if os.path.isdir(os.path.join(train, "data")) else 0,
                          "files": walk(args.data_dir, os.path.dirname(args.data_dir))},
        "fine_tuning": "KHÔNG có checkpoint/adapter nào được huấn luyện trong bài nộp này trừ khi tồn tại /code/models/lora (xem training_code/README.md)",
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=1)
    print("đã ghi", args.out, "|", len(manifest["base_model"]["files"]), "tệp model |", len(manifest["training_data"]["files"]), "tệp dữ liệu")


if __name__ == "__main__":
    main()
