"""Pobiera i czyści dane Common Voice PL, generuje manifesty JSON."""
import argparse
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
from utils.noise import reduce_noise

FS = 16_000


def save_wav(sample, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y = sample["audio"]["array"]
    y = reduce_noise(y, FS)
    fname = out_dir / f"{sample['id']}.wav"
    sf.write(fname, y, FS)
    return str(fname), len(y) / FS


def main(args):
    ds = load_dataset(args.dataset, args.lang, split="train+validation", streaming=False)

    out_audio = Path(args.out) / "wav"
    manifest_train, manifest_valid = [], []

    for i, sample in enumerate(tqdm(ds, desc="🔻 Przetwarzam")):
        wav_path, dur = save_wav(sample, out_audio)
        entry = {"audio_filepath": wav_path, "text": sample["sentence"], "duration": dur}
        if i % 10 == 0:
            manifest_valid.append(entry)
        else:
            manifest_train.append(entry)

    import json, gzip

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.out) / "manifest_train.json.gz", "wt", encoding="utf-8") as f:
        json.dump(manifest_train, f)
    with gzip.open(Path(args.out) / "manifest_valid.json.gz", "wt", encoding="utf-8") as f:
        json.dump(manifest_valid, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="common_voice")
    p.add_argument("--lang", default="pl")
    p.add_argument("--out", required=True)
    main(p.parse_args())