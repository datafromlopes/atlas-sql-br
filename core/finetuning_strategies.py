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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    AdaLoraConfig,
    TaskType
)
import polars as pl
from absl import app, flags
import numpy as np
from utils import (
    BASE_MODEL,
    EPOCHS,
    BATCH_SIZE,
    GRAD_ACCUM,
    MODELS_PATH
)
from utils import GeoDataset
import gc
import os
import json
import math
from utils import Logger

# =======================
# 0. SETTINGS
# =======================

_device = None

if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")

def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

logger = Logger(pid_name="Fine-Tuning Models").setup_logging()

# =============================================================================================
# DATA PREPARATION
# =============================================================================================

class Text2SQLDataset(Dataset):
    def __init__(self, data: pl.DataFrame, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)

        # Input: question + context
        input_text = f"Traduza para SQL: {row['question']}"
        if row['territorial_division']:
            input_text += f" [Divisão: {row['territorial_division']}]"
        if row['geospatial_functions']:
            input_text += f" [Funções: {row['geospatial_functions']}]"

        # Output: SQL
        target_text = row['sql_code']

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


class OptimizedDataCollator:
    def __init__(self, tokenizer, model=None, padding=True):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding

    def __call__(self, features):
        input_ids = np.array([f['input_ids'].numpy() for f in features])
        attention_mask = np.array([f['attention_mask'].numpy() for f in features])
        labels = np.array([f['labels'].numpy() for f in features])

        batch = {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'labels': torch.from_numpy(labels)
        }

        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        return batch



def prepare_data(df: pl.DataFrame, tokenizer, train_split=0.8):
    n_train = int(len(df) * train_split)

    train_data = df[:n_train]
    val_data = df[n_train:]

    train_dataset = Text2SQLDataset(train_data, tokenizer)
    val_dataset = Text2SQLDataset(val_data, tokenizer)

    return train_dataset, val_dataset

# =============================================================================================
# TRAINING MODELS
# =============================================================================================

# =======================
# 1. LORA MODEL
# =======================
def create_lora_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # LoRA Layers
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

# =======================
# 2. ADALORA MODEL
# =======================
def create_adalora_model(total_steps, base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    config = AdaLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        total_step=total_steps,
        target_modules=["q", "v"],
        inference_mode=False
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model

# =======================
# 3. PREFIX-TUNING MODEL
# =======================
def create_prefix_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,
        prefix_projection=True,
        inference_mode=False
    )

    model = get_peft_model(model, prefix_config)
    model.print_trainable_parameters()

    return model

# =======================
# 4. PROMPT TUNING MODEL
# =======================
def create_prompt_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    prompt_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Traduza a pergunta para SQL:",
        tokenizer_name_or_path=base_model_name,
        inference_mode=False
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    return model

# =======================
# 5. PROMPT TUNING V2 MODEL
# =======================
def create_ptuning_v2_model(base_model_name="t5-small", num_virtual_tokens=20):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    prompt_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init="RANDOM",
        inference_mode=False
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    return model

# =======================
# 6. IA3 TUNING MODEL
# =======================
def create_ia3_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    ia3_config = IA3Config(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules=["q", "k", "v", "o", "wi", "wo"]
    )

    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()

    return model

# =======================
# 7. FULL FINE-TUNING MODEL
# =======================
def create_full_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    return model

# =======================
# 8. PYTORCH MODEL - FROM SCRATCH
# =======================
class SimpleSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            embed_dim,
            hidden_dim * 2,
            num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, target_ids=None):
        # Encoder
        embedded = self.dropout(self.embedding(input_ids))
        encoder_output, (hidden, cell) = self.encoder(embedded)

        if target_ids is not None:
            # Training mode
            target_embedded = self.dropout(self.embedding(target_ids))
            decoder_output, _ = self.decoder(target_embedded, (hidden, cell))
            logits = self.output_layer(decoder_output)
            return logits
        else:
            # Inference mode (simplified)
            return encoder_output

# =============================================================================================
# TRAINING FUNCTIONS
# =============================================================================================
def train_from_scratch(train_dataset, val_dataset, tokenizer, epochs=3):
    vocab_size = len(tokenizer)
    model = SimpleSeq2SeqModel(vocab_size)

    global _device
    model.to(_device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(_device)
            labels = batch['labels'].to(_device)

            optimizer.zero_grad()

            logits = model(input_ids, labels[:, :-1])

            loss = criterion(
                logits.reshape(-1, vocab_size),
                labels[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    return model

def train_model(
        model,
        train_dataset,
        val_dataset,
        tokenizer,
        output_dir,
        load_best_model_at_end=False
):
    global _device
    model.to(_device)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        gradient_checkpointing=False,
        load_best_model_at_end=load_best_model_at_end,
        report_to="none"
    )

    data_collator = OptimizedDataCollator(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train()

    return trainer

def save_training_artifacts(
        trainer: Trainer,
        tokenizer,
        output_dir: str,
        save_best: bool = False,
        metadata: dict | None = None
):
    model_to_save = (trainer.model if not save_best else trainer.model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if trainer.state.log_history:
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

# =============================================================================================
# MAIN PIPELINE
# =============================================================================================
def main(argv):
    del argv
    df = GeoDataset.get_dataset().collect()

    print(f"Dataset: {df.shape}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset, val_dataset = prepare_data(df, tokenizer)

    finetuning_models = [
        'LORA',
        'ADALORA',
        'PREFIX',
        'PROMPT',
        'IA3',
        'FULL'
    ]

    for finetuning_model in finetuning_models:
        clear_memory()
        logger.banner(f"STARTING {finetuning_model} TRAINING")
        logger.info("Loading model")

        model = None
        match finetuning_model:
            case 'LORA':
                model = create_lora_model(BASE_MODEL)
            case 'ADALORA':
                total_steps = math.ceil(
                    len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM)
                ) * EPOCHS
                model = create_adalora_model(total_steps, BASE_MODEL)
            case 'PREFIX':
                model = create_prefix_model(BASE_MODEL)
            case 'PROMPT':
                model = create_prompt_model(BASE_MODEL)
            case 'IA3':
                model = create_ia3_model(BASE_MODEL)
            case 'FULL':
                model = create_full_model(BASE_MODEL)

        load_best_model_at_end = False
        if finetuning_model in ['LORA', 'ADALORA', 'FULL']:
            load_best_model_at_end = True

        logger.section("Training")
        trainer = train_model(
            model,
            train_dataset,
            val_dataset,
            tokenizer,
            f"{MODELS_PATH}/{finetuning_model.lower()}",
            load_best_model_at_end
        )
        logger.info("Saving")
        save_training_artifacts(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=f"{MODELS_PATH}/{finetuning_model.lower()}/final",
            metadata={
                "finetuning_type": finetuning_model.lower(),
                "base_model": BASE_MODEL,
                "epochs": EPOCHS
            }
        )
        del model, trainer

    # =======================
    # EXPERIMENT 6: Model from Scratch
    # =======================
    # model_scratch = train_from_scratch(train_dataset, val_dataset, tokenizer, EPOCHS)
    # torch.save(model_scratch.state_dict(), "./models/from_scratch/model.pt")
    # del model_scratch
    # clear_memory()

    print("\n=== Treinamento completo! ===")

if __name__ == "__main__":
    app.run(main)