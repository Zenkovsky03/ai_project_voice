import argparse
import tkinter as tk
import numpy as np
import threading
import queue
import sounddevice as sd
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from utils.noise import reduce_noise

SAMPLE_RATE = 16_000
audio_queue = queue.Queue()
recording = False
audio_frames = []

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
    global recording, audio_frames
    asr, translator = load_models(args.model)

    stream = None

    root = tk.Tk()
    root.title("PL ▶ EN ASR")
    root.geometry("520x260")

    pl_var, en_var = tk.StringVar(), tk.StringVar()
    is_recording = tk.BooleanVar(value=False)

    def audio_callback(indata, frames, time, status):
        if recording:
            audio_frames.append(indata.copy())

    def start_recording():
        nonlocal stream
        audio_frames.clear()
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
        stream.start()
        pl_var.set("🎙️ Nagrywam...")
        en_var.set("")
        btn.config(text="⏹️ Zatrzymaj nagrywanie")
        is_recording.set(True)

    def stop_recording():
        nonlocal stream
        if stream:
            stream.stop()
            stream.close()
        btn.config(text="⏳ Przetwarzam...", state="disabled")
        is_recording.set(False)
        process_audio()

    def process_audio():
        raw_audio = np.concatenate(audio_frames).squeeze()
        pl_var.set("⏳ Przetwarzam...")
        en_var.set("...")

        def run():
            try:
                pl, en = transcribe_translate(asr, translator, raw_audio)
                pl_var.set(pl)
                en_var.set(en)
            except Exception as e:
                pl_var.set("[BŁĄD] " + str(e))
                en_var.set("")
            btn.config(text="🎙️ Rozpocznij nagrywanie", state="normal")

        threading.Thread(target=run).start()

    def on_click():
        global recording
        if not is_recording.get():
            recording = True
            start_recording()
        else:
            recording = False
            stop_recording()

    btn = tk.Button(root, text="🎙️ Rozpocznij nagrywanie", command=on_click, font=("Arial", 27))
    btn.pack(pady=10)

    tk.Label(root, text="🇵🇱", font=("Arial", 27)).pack()
    tk.Label(root, textvariable=pl_var, wraplength=500, font=("Arial", 15)).pack(pady=5)
    tk.Label(root, text="🇬🇧", font=("Arial", 27)).pack()
    tk.Label(root, textvariable=en_var, fg="blue", wraplength=500, font=("Arial", 15)).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="checkpoint folder lub nazwa modelu, np. 'small'")
    main(ap.parse_args())
