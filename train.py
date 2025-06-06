"""Fine‑tuning OpenAI Whisper za pomocą LoRA (PEFT)."""
import argparse, json, gzip, os
from pathlib import Path

import torch
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,  # Zmiana 1: używamy Seq2SeqTrainingArguments
    GenerationConfig,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

NAME = "openai/whisper-small"  # można zmienić na "tiny" dla słabszych GPU


def load_manifest(manifest_path):
    with gzip.open(manifest_path, "rt", encoding="utf-8") if str(manifest_path).endswith(".gz") else open(manifest_path) as f:
        return json.load(f)


# Zmiana 2: Dodajemy data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(entries, processor):
    ds = Dataset.from_list(entries)

    def _prepare(batch):
        path = batch["audio_filepath"]
        audio_array, sample_rate = sf.read(path)

        # Upewnij się, że audio ma odpowiednią częstotliwość próbkowania
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        # Przetwórz audio - WAŻNE: zwróć jako numpy array, nie tensor
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np"  # Zmienione na numpy
        ).input_features[0]

        batch["input_features"] = input_features

        # Tokenizuj tekst - też jako numpy/lista
        labels = processor.tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=448
        ).input_ids

        batch["labels"] = labels

        return batch

    # Użyj multiprocessing dla przyspieszenia z Ryzen 7 7700
    return ds.map(_prepare, remove_columns=ds.column_names, num_proc=1)


def main(args):
    print("🚀 Rozpoczynam przygotowanie modelu...")

    # Sprawdź dostępność CUDA i optymalizacji
    print(f"🔧 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability()}")

        # Włącz optymalizacje CUDA
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 włączone")

    # Zmiana 3: Ustawienie procesora z odpowiednimi parametrami
    processor = WhisperProcessor.from_pretrained(
        NAME,
        language="polish",  # zmienione na małe litery
        task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Użyj fp16
        device_map="auto" if torch.cuda.is_available() else None,  # Auto device mapping
    )
    print(f"✅ Załadowano model: {NAME}")

    # Zmiana 4: Agresywniejsza konfiguracja LoRA dla szybkości
    lora_config = LoraConfig(
        r=64,  # Zwiększone dla lepszej wydajności
        lora_alpha=128,  # Zwiększone proporcjonalnie
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Wydrukuj informacje o parametrach
    print("📊 Parametry modelu:")
    model.print_trainable_parameters()

    print("📁 Ładowanie danych...")
    train_entries = load_manifest(args.train_manifest)
    valid_entries = load_manifest(args.valid_manifest)
    print(f"   Training samples: {len(train_entries)}")
    print(f"   Validation samples: {len(valid_entries)}")

    # Użyj więcej danych - twój sprzęt to udźwignie
    train_size = min(5000, len(train_entries))  # Znacznie zwiększone
    valid_size = min(1000, len(valid_entries))   # Znacznie zwiększone

    print(f"🔄 Przetwarzanie {train_size} próbek treningowych (wielowątkowo)...")
    train_ds = prepare_dataset(train_entries[:train_size], processor)
    print(f"🔄 Przetwarzanie {valid_size} próbek walidacyjnych (wielowątkowo)...")
    valid_ds = prepare_dataset(valid_entries[:valid_size], processor)

    print("✅ Dataset przygotowany!")

    # Zmiana 5: Używamy Seq2SeqTrainingArguments z odpowiednimi parametrami
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=32,  # Zwiększony batch size dla RTX 4070 Super
        gradient_accumulation_steps=4,   # Zmniejszony bo większy batch size
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_steps=200,
        num_train_epochs=3,
        learning_rate=5e-5,  # Zwiększony LR bo większy batch
        warmup_steps=100,
        fp16=True,  # Wymuszone fp16 dla prędkości
        dataloader_num_workers=0,  # Wykorzystaj Ryzen 7 7700
        generation_max_length=225,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=True,  # Włącz dla GPU
        report_to=None,
        logging_first_step=True,
        save_safetensors=True,
        # Optymalizacje dla RTX 4070 Super:
        tf32=True,  # Włącz TensorFloat-32 dla Ampere/Ada
        gradient_checkpointing=False,  # Wyłącz - masz dość VRAM
        optim="adamw_torch_fused",  # Szybszy optimizer
        lr_scheduler_type="cosine",  # Lepszy scheduler
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Batch size auto-finding (opcjonalne):
        auto_find_batch_size=True,  # Znajdzie maksymalny batch size
    )

    # Zmiana 7: Utworzenie data collatora
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,  # MUSISZ używać data collatora
        processing_class=processor.feature_extractor,  # Zmienione z tokenizer na processing_class
    )

    # Rozpocznij trening
    print("🏋️ Rozpoczynam trening...")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Total steps: {len(train_ds) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

    try:
        trainer.train()
        print("🎉 Trening zakończony pomyślnie!")
    except Exception as e:
        print(f"❌ Błąd podczas treningu: {e}")
        raise

    # Zapisz model i procesor
    print("💾 Zapisywanie modelu...")
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    print(f"Model zapisany w: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", required=True, help="Ścieżka do manifestu treningowego")
    p.add_argument("--valid_manifest", required=True, help="Ścieżka do manifestu walidacyjnego")
    p.add_argument("--out_dir", default="checkpoints/whisper-lora-pl", help="Katalog wyjściowy")
    main(p.parse_args())