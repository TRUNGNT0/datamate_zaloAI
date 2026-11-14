"""Backend local bằng transformers (+ bitsandbytes 4-bit). Chậm hơn vLLM nhưng ổn định, chạy được trên T4/P100."""
from __future__ import annotations

import os
import re

from datamate.config import Config
from .base import GenerationError, LLMBackend
from .local_common import (has_module, is_hybrid_model, is_prequantized, quant_method, resolve_model_path,
                           sampling_kwargs, use_4bit, weights_gb, read_text_config)


def hf_model_path(cfg: Config, path: str) -> str:
    """Engine hf không đọc được AWQ/GPTQ nếu thiếu thư viện -> dùng model gốc fp16 (nén 4-bit khi nạp)."""
    qm = quant_method(path)
    need = {"awq": ("awq",), "gptq": ("gptqmodel", "auto_gptq")}.get(qm)
    if need is None or has_module(*need):
        return path
    if cfg.hf_base_model:
        base = cfg.hf_base_model
    elif not os.path.isdir(cfg.local_model):
        base = re.sub(r"-(AWQ|GPTQ.*)$", "", cfg.local_model, flags=re.I)
    else:
        raise RuntimeError(f"Model {qm.upper()} cần thư viện tương ứng khi chạy engine hf. Đặt hf_base_model = repo "
                           "model gốc fp16 (vd 'Qwen/Qwen2.5-Coder-7B-Instruct') hoặc chạy engine vllm.")
    print(f"Engine hf không đọc được {qm.upper()} -> dùng model gốc {base} (nén 4-bit khi nạp).")
    return resolve_model_path(base)


def prefill_chunked(model, ids, chunk: int):
    """Nạp prompt vào KV-cache từng đoạn `chunk` token (chừa token cuối cho generate). Attention mỗi đoạn chỉ
    chunk×L thay vì L×L (prompt ~9k token cần ~8,7 GiB ma trận attention, nổ VRAM T4)."""
    import torch
    from transformers import DynamicCache
    cache, last = DynamicCache(), ids.shape[1] - 1
    with torch.no_grad():
        for start in range(0, last, chunk):
            model(input_ids=ids[:, start:min(start + chunk, last)], past_key_values=cache, use_cache=True)
    return cache


class HFBackend(LLMBackend):
    supports_batch = True
    engine = "hf"

    def __init__(self, cfg: Config, path: str, n_gpu: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self._torch = cfg, torch
        path = hf_model_path(cfg, path)
        self.tok = AutoTokenizer.from_pretrained(path)
        use_cuda = n_gpu > 0
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if use_cuda else 0.0
        quant4 = use_cuda and not is_prequantized(path) and use_4bit(cfg.hf_quant, vram_gb, weights_gb(path))
        self.quant = "4bit" if quant4 else "full"
        self.hybrid = is_hybrid_model(path)   # Qwen3.5: không dùng prefill_chunked (cache lai linear-attention)
        kw: dict = {"attn_implementation": "sdpa"}
        if use_cuda:
            kw["device_map"] = {"": 0}        # 1 GPU: 7B 4-bit chỉ ~5,5GB; chia qua 2 T4 chỉ thêm độ trễ
            if quant4:                        # model gốc fp16 -> nén 4-bit NF4 khi nạp (khớp cách QLoRA huấn luyện)
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
        if not use_cuda:
            dtype = torch.float32
        elif quant4 or is_prequantized(path) or not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16            # Qwen3.5 huấn luyện bằng bf16; fp16 dễ tràn số
        print(f"[hf] {path} | GPU {vram_gb:.0f}GB | nạp {self.quant} | {dtype} | lai linear-attention: {self.hybrid}")
        model_class = AutoModelForCausalLM
        if read_text_config(path).get("model_type") == "qwen3_5_text":
            # Official text-only loader also accepts the composite Qwen3.5 checkpoint.
            from transformers import Qwen3_5ForCausalLM
            model_class = Qwen3_5ForCausalLM
        try:
            model = model_class.from_pretrained(path, dtype=dtype, **kw)
        except TypeError:
            model = model_class.from_pretrained(path, torch_dtype=dtype, **kw)
        if cfg.local_lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.local_lora)
            if not quant4:                    # nạp đầy đủ thì gộp LoRA vào trọng số: sinh nhanh hơn, không còn nhánh LoRA mỗi lớp
                model = model.merge_and_unload()
        self.model = model.eval()

    def tag(self) -> str:
        c = self.cfg
        return f"local|{c.local_model}|{c.local_lora}|hf-{self.quant}|T{c.temperature}|L{c.max_model_len}"

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate(self, system: str, prompt: str, n: int = 1) -> list[str]:
        return self.generate_with_options(system, prompt, n=n)

    def prompt_tokens(self, system: str, prompt: str) -> int:
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, **self.cfg.chat_kwargs)
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate_with_options(self, system: str, prompt: str, n: int = 1,
                              max_new_tokens: int | None = None, temperature: float | None = None) -> list[str]:
        """n lời giải trong MỘT lượt (prefill một lần rồi nhân bản cache); hết VRAM thì tách lẻ từng mẫu."""
        torch, cfg, tok = self._torch, self.cfg, self.tok
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        prompt_len = ids.shape[1]
        output_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        temp = cfg.temperature if temperature is None else temperature
        if prompt_len + output_tokens > cfg.max_model_len:
            raise GenerationError(f"Context too long: {prompt_len} prompt + {output_tokens} output > {cfg.max_model_len}")

        def run(k: int) -> list[str]:
            sample = (dict(do_sample=True, **sampling_kwargs(cfg.local_model, temp))
                      if temp > 0 else dict(do_sample=False))
            x, extra = ids, {}
            if cfg.prefill_chunk and not self.hybrid:
                cache = prefill_chunked(self.model, ids, cfg.prefill_chunk)
                if k > 1:
                    cache.batch_repeat_interleave(k)
                extra["past_key_values"] = cache
            if k > 1:
                x = ids.repeat(k, 1)
            with torch.no_grad():
                out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                          max_new_tokens=output_tokens, pad_token_id=tok.eos_token_id,
                                          **extra, **sample)
            return [tok.decode(o[prompt_len:], skip_special_tokens=True) for o in out]

        try:
            try:
                return run(n)
            except Exception as e:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                print(f"[local] sinh batch {n} lỗi ({type(e).__name__}: {str(e)[:120]}) -> sinh lần lượt từng mẫu")
                return [run(1)[0] for _ in range(n)]
        except Exception as e:
            torch.cuda.empty_cache()                 # thường là OOM khi prompt quá dài
            raise GenerationError(repr(e)[:300]) from e
