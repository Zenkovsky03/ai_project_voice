"""Aplikacja Tkinter: nagrywanie ▶ transkrypcja PL ▶ tłumaczenie EN."""
import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd
from transformers import pipeline
import whisper
from utils.noise import reduce_noise
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import pipeline
import threading

SAMPLE_RATE = 16_000


def record(seconds=5):
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()


def load_models(model_path: str):
    processor = WhisperProcessor.from_pretrained('./checkpoints/whisper-lora-pl')
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
    return (model, processor), translator


def transcribe_translate(asr_tuple, translator, audio):
    model, processor = asr_tuple
    audio = reduce_noise(audio, SAMPLE_RATE)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(inputs.input_features, max_length=448)
    pl_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    en_text = translator(pl_text, max_length=256)[0]['translation_text']
    return pl_text, en_text


def main(args):
    asr, translator = load_models(args.model)

    root = tk.Tk(); root.title("PL ▶ EN ASR")
    root.geometry("520x260")

    pl_var, en_var = tk.StringVar(), tk.StringVar()

    def run_transcription():
        btn.config(text="🎙️ Nagrywam...", state="disabled")
        pl_var.set("(nagrywam...)")
        en_var.set("")
        root.update_idletasks()

        # 🔴 Nagrywanie
        raw_audio = record(args.seconds)

        # 🔄 Przed transkrypcją - wróć do trybu oczekiwania
        btn.config(text="⏳ Przetwarzam...", state="disabled")
        pl_var.set("⏳ Przetwarzam nagranie...")
        en_var.set("...")
        root.update_idletasks()

        # 🧠 Transkrypcja i tłumaczenie
        pl, en = transcribe_translate(asr, translator, raw_audio)

        # 🟢 Wynik + przywrócenie przycisku
        pl_var.set(pl)
        en_var.set(en)
        btn.config(text=f"🎙️ Nagraj {args.seconds} s", state="normal")

    def on_click():
        threading.Thread(target=run_transcription).start()

    btn = tk.Button(root, text=f"🎙️ Nagraj {args.seconds} s", command=on_click, font=("Arial", 14))
    btn.pack(pady=10)

    tk.Label(root, text="🇵🇱", font=("Arial", 12)).pack()
    tk.Label(root, textvariable=pl_var, wraplength=500, font=("Arial", 10)).pack(pady=5)
    tk.Label(root, text="🇬🇧", font=("Arial", 12)).pack()
    tk.Label(root, textvariable=en_var, fg="blue", wraplength=500, font=("Arial", 10)).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="checkpoint folder lub nazwa modelu, np. 'small'")
    ap.add_argument("--seconds", type=int, default=5, help="długość nagrania")
    main(ap.parse_args())