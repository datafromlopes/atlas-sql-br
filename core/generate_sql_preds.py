# coding=utf-8
# Copyright (C) 2026  Diego Lopes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
from __future__ import annotations

import gc
import json
from pathlib import Path
import traceback

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    BASE_MODEL,
    ARTIFACT_PATH,
    INPUT_FILE,
    OUTPUT_FILE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def clear_memory() -> None:
    """Free cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tokenizer():
    print(f"[INFO] Loading tokenizer from : {BASE_MODEL}")
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def load_base_model():
    """Load the base LLaMA model."""
    clear_memory()
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model


def load_finetuned_model():
    """Load the PEFT fine-tuned model on top of a fresh base instance."""
    clear_memory()
    print(f"[INFO] Loading fine-tuned model from : {ARTIFACT_PATH}")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ),
        ARTIFACT_PATH,
    )
    model.eval()
    return model


def unload_model(model) -> None:
    """Delete model object and free GPU memory."""
    del model
    clear_memory()
    print("[INFO] Model unloaded, GPU memory freed.\n")


def predict_sql(tokenizer, model, question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return _clean_sql(tokenizer.decode(generated, skip_special_tokens=True))


def _clean_sql(text: str) -> str:
    """Strip markdown fences if the model wraps the output in them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def run_pass(
    tokenizer, model, validation_data: list[dict], label: str
) -> dict[int, str]:
    total = len(validation_data)
    predictions: dict[int, str] = {}

    print(f"\n{'=' * 60}")
    print(f"  {label}  ({total} questions)")
    print(f"{'=' * 60}")

    for idx, item in enumerate(validation_data, start=1):
        validation_id = item["id"]
        question      = item["question"]

        print(
            f"[{idx:>4}/{total}] id={validation_id:>4}  {question[:70]}...",
            end=" ", flush=True,
        )

        try:
            predictions[validation_id] = predict_sql(tokenizer, model, question)
            print("OK")
        except Exception as exc:
            predictions[validation_id] = ""
            print(f"ERRO  {exc}")
            traceback.print_exc()  # imprime o traceback completo

    return predictions


# ========================================================
# Main
# ========================================================

def main() -> None:
    input_path  = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    with input_path.open(encoding="utf-8") as f:
        validation_data: list[dict] = json.load(f)

    total = len(validation_data)
    print(f"[INFO] {total} questions loaded from {input_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = load_tokenizer()

    # ── Pass 1: base model ───────────────────────────────────────────────────
    base_model = load_base_model()
    base_preds = run_pass(tokenizer, base_model, validation_data, "Pass 1 - base model")
    unload_model(base_model)

    # ── Pass 2: fine-tuned model ─────────────────────────────────────────────
    finetuned_model = load_finetuned_model()
    finetuned_preds = run_pass(tokenizer, finetuned_model, validation_data, "Pass 2 - fine-tuned model")
    unload_model(finetuned_model)

    # ── Merge and save ───────────────────────────────────────────────────────
    results = [
        {
            "sql_validation_id":  item["id"],
            "sql_code_base":      base_preds.get(item["id"], ""),
            "sql_code_finetuned": finetuned_preds.get(item["id"], ""),
        }
        for item in validation_data
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(results)} predictions saved -> {output_path.resolve()}")


if __name__ == "__main__":
    main()