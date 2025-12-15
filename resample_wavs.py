import argparse
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: Path):
    audio, sr = sf.read(path, always_2d=False)
    return audio, sr


def save_audio(path: Path, audio, sr: int, subtype: str = "PCM_16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype=subtype)


def resample_audio(audio, orig_sr: int, target_sr: int):
    if orig_sr == target_sr:
        return audio
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down)


def process_dataset(src_root: Path, dst_root: Path, target_sr: int = 22050):
    wavs = list(src_root.rglob("*.wav"))
    total = len(wavs)
    converted = 0
    skipped = 0

    for i, wav_path in enumerate(wavs, 1):
        rel = wav_path.relative_to(src_root)
        out_path = dst_root / rel

        audio, sr = load_audio(wav_path)

        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            converted += 1
        else:
            skipped += 1

        save_audio(out_path, audio, target_sr)

        if i % 500 == 0 or i == total:
            print(f"[{i}/{total}] готово (конвертировано: {converted}, без изменений: {skipped})")

    print(f"\nИтог: всего {total}, конвертировано {converted}, без изменений {skipped}.")
    print(f"Результат: {dst_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Ресемплинг всех WAV до нужной частоты."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data"),
        help="Каталог с исходными WAV (по умолчанию data)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data_22050"),
        help="Каталог для сохранения результата (по умолчанию data_22050)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Целевая частота дискретизации (по умолчанию 22050)",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Источник не найден: {args.src}")

    process_dataset(args.src, args.dst, args.sr)


if __name__ == "__main__":
    main()
