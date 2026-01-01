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
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    PeftModel
)
import polars as pl
from typing import Dict, List
from absl import app, flags
import os

from utils import (
    BASE_MODEL,
    EPOCHS
)
from utils import GeoDataset

# =======================
# 1. DATA PREPARATION
# =======================

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


def prepare_data(df: pl.DataFrame, tokenizer, train_split=0.8):
    n_train = int(len(df) * train_split)

    train_data = df[:n_train]
    val_data = df[n_train:]

    train_dataset = Text2SQLDataset(train_data, tokenizer)
    val_dataset = Text2SQLDataset(val_data, tokenizer)

    return train_dataset, val_dataset

# =======================
# 2. LORA MODEL
# =======================

def create_lora_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]  # LoRA Layers
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# =======================
# 3. PREFIX-TUNING MODEL
# =======================

def create_prefix_tuning_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,
        prefix_projection=True
    )

    model = get_peft_model(model, prefix_config)
    model.print_trainable_parameters()

    return model


# =======================
# 4. PROMPT TUNING MODEL
# =======================

def create_prompt_tuning_model(base_model_name="t5-small", tokenizer=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    prompt_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Traduza a pergunta para SQL:",
        tokenizer_name_or_path=base_model_name
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    return model


# =======================
# 5. ADAPTER MODEL
# =======================

class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x + residual


def add_adapters_to_model(model, adapter_size=64):
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.d_model

    for i, layer in enumerate(model.encoder.block):
        adapter = AdapterLayer(hidden_size, adapter_size)
        layer.adapter = adapter

        original_forward = layer.forward

        def forward_with_adapter(self, hidden_states, *args, **kwargs):
            output = original_forward(hidden_states, *args, **kwargs)
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = self.adapter(hidden_states)
                return (hidden_states,) + output[1:]
            else:
                return self.adapter(output)

        layer.forward = lambda *args, s=layer, **kwargs: forward_with_adapter(s, *args, **kwargs)

    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'adapter' in name:
            param.requires_grad = True
            trainable_params += param.numel()

    return model


# =======================
# 6. FULL FINE-TUNING MODEL
# =======================

def create_full_finetuning_model(base_model_name="t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return model


# =======================
# 7. PYTORCH MODEL - FROM SCRATCH
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


def train_from_scratch(train_dataset, val_dataset, tokenizer, epochs=3):
    vocab_size = len(tokenizer)
    model = SimpleSeq2SeqModel(vocab_size)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

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

# =======================
# 8. TRAINING FUNCTIONS
# =======================

def train_model(model, train_dataset, val_dataset, tokenizer, output_dir, epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train()

    return trainer

# =======================
# 9. MAIN PIPELINE
# =======================

def main(argv):
    del argv
    df = GeoDataset.get_dataset()

    print(f"Dataset: {df.shape}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_dataset, val_dataset = prepare_data(df, tokenizer)

    # =======================
    # EXPERIMENT 1: LoRA
    # =======================
    model_lora = create_lora_model(BASE_MODEL)
    train_model(model_lora, train_dataset, val_dataset, tokenizer, "./models/lora", EPOCHS)

    # =======================
    # EXPERIMENT 2: Prefix-Tuning
    # =======================
    model_prefix = create_prefix_tuning_model(BASE_MODEL)
    train_model(model_prefix, train_dataset, val_dataset, tokenizer, "./models/prefix", EPOCHS)

    # =======================
    # EXPERIMENT 3: Prompt Tuning
    # =======================
    model_prompt = create_prompt_tuning_model(BASE_MODEL, tokenizer)
    train_model(model_prompt, train_dataset, val_dataset, tokenizer, "./models/prompt", EPOCHS)

    # =======================
    # EXPERIMENT 4: Adapter
    # =======================
    model_adapter = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model_adapter = add_adapters_to_model(model_adapter)
    train_model(model_adapter, train_dataset, val_dataset, tokenizer, "./models/adapter", EPOCHS)

    # =======================
    # EXPERIMENT 5: Full Fine-tuning
    # =======================
    model_full = create_full_finetuning_model(BASE_MODEL)
    train_model(model_full, train_dataset, val_dataset, tokenizer, "./models/full_finetuning", EPOCHS)

    # =======================
    # EXPERIMENT 6: Model from Scratch
    # =======================
    model_scratch = train_from_scratch(train_dataset, val_dataset, tokenizer, EPOCHS)
    torch.save(model_scratch.state_dict(), "./models/from_scratch/model.pt")

    print("\n=== Treinamento completo! ===")
    print("Modelos salvos em:")
    print("  - ./models/lora")
    print("  - ./models/prefix")
    print("  - ./models/prompt")
    print("  - ./models/adapter")
    print("  - ./models/full_finetuning")
    print("  - ./models/from_scratch")

if __name__ == "__main__":
    app.run(main)