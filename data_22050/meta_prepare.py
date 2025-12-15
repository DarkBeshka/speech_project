import csv
import random
from pathlib import Path

RUSLAN_ROOT = Path("data")
SRC_META = RUSLAN_ROOT / "metadata_RUSLAN_22200.csv"
OUT_ALL = RUSLAN_ROOT / "ruslan_meta.txt"
OUT_TRAIN = RUSLAN_ROOT / "metadata_train.txt"
OUT_VAL = RUSLAN_ROOT / "metadata_val.txt"


def detect_delimiter(sample_line: str):
    candidates = ["\t", ";", "|", ","]
    counts = {d: sample_line.count(d) for d in candidates}
    # choose delimiter with max occurrences (fallback to '|')
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "|"
    return best


rows = []
with open(SRC_META, "r", encoding="utf-8") as f:
    first = f.readline()
    if not first:
        raise SystemExit(f"Empty file: {SRC_META}")
    delim = detect_delimiter(first)
    # detect header-like first line
    header_tokens = [t.strip().lower() for t in first.split(delim)]
    has_header = any(
        any(k in t for k in ("file", "id", "text", "transcript", "sentence"))
        for t in header_tokens
    )

    f.seek(0)
    if has_header:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            try:
                # try common column names first
                file_id = None
                text = None
                # direct keys
                if "file_id" in row:
                    file_id = row.get("file_id")
                if "text" in row:
                    text = row.get("text")

                # fallback: find columns by keyword
                if not file_id:
                    for k in row:
                        if k and ("file" in k.lower() or "id" in k.lower()):
                            file_id = row[k]
                            break
                if not text:
                    for k in row:
                        if k and ("text" in k.lower() or "transcript" in k.lower() or "sentence" in k.lower()):
                            text = row[k]
                            break

                if file_id is None or text is None:
                    continue
                file_id = str(file_id).strip()
                text = str(text).strip()
            except Exception:
                continue
            if file_id.endswith(".wav"):
                file_id = file_id[:-4]
            if len(text) < 3:
                continue
            rows.append((file_id, text))
    else:
        # headerless: parse each line splitting into two parts
        for line in f:
            if not line.strip():
                continue
            parts = line.split(delim, 1)
            if len(parts) < 2:
                # try whitespace split as last resort
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
            file_id, text = parts[0].strip(), parts[1].strip()
            if file_id.endswith(".wav"):
                file_id = file_id[:-4]
            if len(text) < 3:
                continue
            rows.append((file_id, text))

print(f"Всего предложений после фильтрации: {len(rows)}")

random.shuffle(rows)
n_total = len(rows)
n_val = max(500, int(0.05 * n_total)) if n_total > 0 else 0

val_rows = rows[:n_val]
train_rows = rows[n_val:]


def write_meta(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for file_id, text in data:
            f.write(f"{file_id}|{text}\n")


write_meta(OUT_ALL, rows)
write_meta(OUT_TRAIN, train_rows)
write_meta(OUT_VAL, val_rows)

print("Готово:")
print(" -", OUT_ALL)
print(" -", OUT_TRAIN)
print(" -", OUT_VAL)