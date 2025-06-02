"""CLI: podaj plik audio (WAV/FLAC/MP3) → transkrypcja PL i opcjonalnie tłumaczenie EN."""
import argparse, soundfile as sf, whisper
from transformers import pipeline
from utils.noise import reduce_noise

EXPECTED_SR = 16_000

def main(a):
    # 1️⃣ Wczytaj audio
    audio, sr = sf.read(a.audio)
    if sr != EXPECTED_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=EXPECTED_SR)
        except ImportError:
            raise RuntimeError("Plik ma %s Hz — zainstaluj librosa lub podaj 16 kHz" % sr)
        sr = EXPECTED_SR
    audio = audio.astype('float32')
    audio = reduce_noise(audio, sr)

    # 2️⃣ Modele
    asr = whisper.load_model(a.model)
    result = asr.transcribe(audio, language='pl', task='transcribe')
    pl_text = result['text'].strip()

    print("PL:", pl_text)

    if a.translate:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
        en_text = translator(pl_text)[0]['translation_text']
        print("EN:", en_text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="ścieżka do pliku WAV/FLAC/MP3 (mono lub stereo)")
    p.add_argument("--model", default="small", help="checkpoint Whispera lub 'small'/'tiny'…")
    p.add_argument("--translate", action="store_true", help="również tłumacz PL→EN")
    main(p.parse_args())