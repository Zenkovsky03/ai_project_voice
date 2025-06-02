"""Aplikacja Tkinter: nagrywanie ▶ transkrypcja PL ▶ tłumaczenie EN."""
import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd
from transformers import pipeline
import whisper
from utils.noise import reduce_noise

SAMPLE_RATE = 16_000


def record(seconds=5):
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()


def load_models(model_name: str):
    asr = whisper.load_model(model_name)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
    return asr, translator


def transcribe_translate(asr, translator, audio):
    audio = reduce_noise(audio, SAMPLE_RATE)
    result = asr.transcribe(audio, language='pl', task='transcribe')
    pl_text = result['text'].strip()
    en_text = translator(pl_text, max_length=256)[0]['translation_text']
    return pl_text, en_text


def main(args):
    asr, translator = load_models(args.model)

    root = tk.Tk(); root.title("PL ▶ EN ASR")
    root.geometry("520x260")

    pl_var, en_var = tk.StringVar(), tk.StringVar()

    def on_click():
        raw_audio = record(args.seconds)
        pl, en = transcribe_translate(asr, translator, raw_audio)
        pl_var.set(pl); en_var.set(en)

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