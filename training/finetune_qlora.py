"""Fine-tune QLoRA tá»« sft_train.jsonl (má»—i dÃ²ng: {"messages": [system, user, assistant]}) rá»“i (tÃ¹y chá»n) gá»™p LoRA vÃ o model gá»‘c.

DÃ¹ng trÃªn Colab. Qwen3.5-9B: A100 náº¡p bf16 Ä‘áº§y Ä‘á»§ + LoRA (--quant auto tá»± chá»n), L4/T4 tá»± nÃ©n 4-bit (QLoRA). CHÆ¯A cháº¡y thá»­
trÃªn GPU: kiá»ƒm báº±ng --dry-run (chá»‰ token hÃ³a, in phÃ¢n bá»‘ Ä‘á»™ dÃ i vÃ  sá»‘ máº«u vá»«a) trÆ°á»›c khi tá»‘n giá» GPU.

    python -m training.finetune_qlora --data sft_train.jsonl --base Qwen/Qwen3.5-9B --out /content/drive/MyDrive/lora --max-len 8192
    python -m training.finetune_qlora --merge --base Qwen/Qwen3.5-9B --out /content/drive/MyDrive/lora --merged /content/merged
"""
from __future__ import annotations

import argparse
import json
import random

from datamate.llm.local_common import is_hybrid_model, is_qwen3_family, resolve_model_path, use_4bit, weights_gb

# Má»i lá»›p tuyáº¿n tÃ­nh cá»§a khá»‘i chÃº Ã½, MLP vÃ  lá»›p linear-attention (Qwen3.5: in_proj_qkv/in_proj_z/out_proj). TÃªn khÃ´ng cÃ³ trong model bá»‹ bá» qua.
# in_proj_a/in_proj_b chá»‰ cÃ³ vÃ i chá»¥c Ä‘áº§u ra nÃªn khÃ´ng gáº¯n LoRA.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


def build_example(tok, messages: list[dict], max_len: int, chat_kwargs: dict | None = None):
    """MÃ£ hÃ³a má»™t máº«u; chá»‰ pháº§n tráº£ lá»i cá»§a assistant Ä‘Æ°á»£c tÃ­nh loss. Máº«u dÃ i hÆ¡n max_len bá»‹ Bá»Ž (khÃ´ng cáº¯t, vÃ¬ cáº¯t
    Ä‘uÃ´i sáº½ lÃ m model há»c code cá»¥t). `chat_kwargs` pháº£i GIá»NG lÃºc suy luáº­n (Qwen3.5: enable_thinking=False), náº¿u khÃ´ng
    pháº§n prompt khÃ¡c tiá»n tá»‘ cá»§a báº£n Ä‘áº§y Ä‘á»§ vÃ  má»i máº«u bá»‹ loáº¡i."""
    kwargs = chat_kwargs or {}
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs)
    full = tok.apply_chat_template(messages, tokenize=False, **kwargs)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def load_examples(tok, path: str, max_len: int, chat_kwargs: dict | None = None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    examples = [e for e in (build_example(tok, r["messages"], max_len, chat_kwargs) for r in rows) if e]
    print(f"{len(examples)}/{len(rows)} máº«u vá»«a max_len={max_len}")
    lengths = sorted(len(e["input_ids"]) for e in examples)
    if lengths:
        print(f"Ä‘á»™ dÃ i token: min {lengths[0]} | trung vá»‹ {lengths[len(lengths) // 2]} | max {lengths[-1]}")
    return examples


def collate(batch, pad_id: int):
    import torch
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], labels[i, :n], mask[i, :n] = (torch.tensor(b["input_ids"]), torch.tensor(b["labels"]),
                                                  torch.ones(n, dtype=torch.long))
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def load_causal_lm(path: str, dtype, **kwargs):
    """Náº¡p model chá»¯-thuáº§n (Qwen3.5 Ä‘a phÆ°Æ¡ng thá»©c váº«n náº¡p Ä‘Æ°á»£c báº±ng AutoModelForCausalLM, bá» pháº§n áº£nh)."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)           # transformers >= 5
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)


def completion_loss_trainer(trainer_cls):
    """Trainer chá»‰ tÃ­nh logits á»Ÿ cÃ¡c vá»‹ trÃ­ cÃ³ nhÃ£n (pháº§n tráº£ lá»i, vÃ i trÄƒm token) thay vÃ¬ cáº£ chuá»—i ~8k token. Tá»« vá»±ng
    Qwen3.5 lÃ  248k nÃªn logits cáº£ chuá»—i (bf16 + báº£n float32 cá»§a loss) tá»‘n cá»¡ 8GB vÃ  lÃ  thá»§ pháº¡m OOM Ä‘áº§u tiÃªn. CHÆ¯A cháº¡y thá»­
    trÃªn GPU; há»ng thÃ¬ cháº¡y láº¡i vá»›i --full-logits (loss chuáº©n cá»§a Trainer)."""

    class CompletionLossTrainer(trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            hidden = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            labels = inputs["labels"][:, 1:]
            keep = labels != -100
            logits = base.lm_head(hidden[:, :-1][keep]).float()
            loss = torch.nn.functional.cross_entropy(logits, labels[keep])
            return (loss, None) if return_outputs else loss

    return CompletionLossTrainer


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    chat_kwargs = {"enable_thinking": False} if is_qwen3_family(args.base) else {}
    tok = AutoTokenizer.from_pretrained(args.base)        # chá»‰ táº£i tokenizer: --dry-run khÃ´ng cáº§n táº£i cáº£ model
    examples = load_examples(tok, args.data, args.max_len, chat_kwargs)
    if args.dry_run or not examples:
        return
    base = resolve_model_path(args.base)
    random.Random(0).shuffle(examples)
    use_bf16 = torch.cuda.is_bf16_supported()          # T4 khÃ´ng cÃ³ bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    quant4 = use_4bit(args.quant, vram_gb, weights_gb(base))
    print(f"[train] GPU {vram_gb:.0f}GB | {'QLoRA 4-bit' if quant4 else 'LoRA trÃªn bf16/fp16 Ä‘áº§y Ä‘á»§'} | "
          f"lai linear-attention: {is_hybrid_model(base)}")
    kw = {"device_map": {"": 0}, "attn_implementation": "sdpa"}
    if quant4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = load_causal_lm(base, dtype, **kw)
    if quant4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    trainer_cls = Trainer if args.full_logits else completion_loss_trainer(Trainer)
    trainer = trainer_cls(
        model=model, train_dataset=examples, data_collator=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
        args=TrainingArguments(
            output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
            bf16=use_bf16, fp16=not use_bf16, gradient_checkpointing=True, logging_steps=5, save_strategy="epoch",
            report_to="none", remove_unused_columns=False, optim="paged_adamw_8bit"))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Ä‘Ã£ lÆ°u LoRA adapter ->", args.out)


def merge(args):
    """Gá»™p adapter vÃ o model gá»‘c (bf16, trÃªn CPU) Ä‘á»ƒ vLLM náº¡p trá»±c tiáº¿p á»Ÿ vÃ²ng ná»™p (khÃ´ng cáº§n peft lÃºc suy luáº­n)."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    path = resolve_model_path(args.base)
    base = load_causal_lm(path, torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
    merged.save_pretrained(args.merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(path).save_pretrained(args.merged)
    print("Ä‘Ã£ lÆ°u model gá»™p ->", args.merged)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="sft_train.jsonl")
    p.add_argument("--base", required=True, help="model Gá»C bf16/fp16 (khÃ´ng pháº£i báº£n AWQ), vd Qwen/Qwen3.5-9B")
    p.add_argument("--out", required=True, help="thÆ° má»¥c lÆ°u LoRA adapter")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto",
                   help="auto: nÃ©n 4-bit (QLoRA) chá»‰ khi VRAM khÃ´ng Ä‘á»§ cho bf16; none: LoRA trÃªn bf16 Ä‘áº§y Ä‘á»§ (A100)")
    p.add_argument("--full-logits", action="store_true",
                   help="dÃ¹ng loss chuáº©n cá»§a Trainer (tÃ­nh logits cáº£ chuá»—i, tá»‘n thÃªm ~8GB á»Ÿ 8k token); máº·c Ä‘á»‹nh chá»‰ tÃ­nh á»Ÿ pháº§n tráº£ lá»i")
    p.add_argument("--dry-run", action="store_true", help="chá»‰ token hÃ³a vÃ  in phÃ¢n bá»‘ Ä‘á»™ dÃ i, khÃ´ng train")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--merged", help="thÆ° má»¥c lÆ°u model Ä‘Ã£ gá»™p (dÃ¹ng vá»›i --merge)")
    args = p.parse_args()
    merge(args) if args.merge else train(args)


if __name__ == "__main__":
    main()

