"""Fine‑tuning OpenAI Whisper za pomocą LoRA (PEFT)."""
import argparse, json, gzip, os
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer

NAME = "openai/whisper-small"  # można zmienić na "tiny" dla słabszych GPU


def load_manifest(manifest_path):
    with gzip.open(manifest_path, "rt", encoding="utf-8") if str(manifest_path).endswith(".gz") else open(manifest_path) as f:
        return json.load(f)


def prepare_dataset(entries, processor):
    ds = load_dataset("json", data_files={"train": entries}, split="train")
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16_000, mono=True))

    def _prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=16_000).input_features[0]
        batch["labels"] = processor(text=batch["text"]).input_ids
        return batch

    return ds.map(_prepare, remove_columns=ds.column_names, num_proc=4)


def main(args):
    processor = WhisperProcessor.from_pretrained(NAME, language="Polish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(NAME)

    # LoRA
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["k_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, lora_config)

    train_entries = load_manifest(args.train_manifest)
    valid_entries = load_manifest(args.valid_manifest)

    train_ds = prepare_dataset(train_entries, processor)
    valid_ds = prepare_dataset(valid_entries, processor)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=processor.feature_extractor,
        data_collator=None,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--valid_manifest", required=True)
    p.add_argument("--out_dir", default="checkpoints/whisper-lora-pl")
    main(p.parse_args())