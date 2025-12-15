import os
import sys
from collections import Counter
from pathlib import Path


def iter_wavs(root: Path):
    for p in root.rglob("*.wav"):
        if p.is_file():
            yield p


def read_sr(wav_path: Path):
    """
    Возвращает sample rate для WAV.
    Пытаемся через soundfile, если нет — через стандартный wave.
    """
    try:
        import soundfile as sf  # type: ignore

        with sf.SoundFile(wav_path) as f:
            return int(f.samplerate)
    except Exception:
        import wave

        with wave.open(str(wav_path), "rb") as f:
            return int(f.getframerate())


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not root.exists():
        print(f"Путь не найден: {root}")
        sys.exit(1)

    counter = Counter()
    mismatches = []
    target_sr = 22050

    for wav_path in iter_wavs(root):
        sr = read_sr(wav_path)
        counter[sr] += 1
        if sr != target_sr:
            mismatches.append((wav_path, sr))

    total = sum(counter.values())
    print(f"Всего wav: {total}")
    for sr, cnt in counter.most_common():
        print(f"SR {sr}: {cnt}")

    if mismatches:
        print("\nФайлы с неподходящей SR (первые 20):")
        for p, sr in mismatches[:20]:
            print(f"{p} -> {sr}")
        if len(mismatches) > 20:
            print(f"... и ещё {len(mismatches) - 20} файлов")
    else:
        print("\nВсе файлы с частотой 22050 Гц.")


if __name__ == "__main__":
    main()