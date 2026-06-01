from datasets import load_dataset
from pathlib import Path
import jsonlines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/qasper/qasper_train.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("allenai/qasper", split="train", trust_remote_code=False)
records = [dict(r) for r in ds.select(range(20))]

with jsonlines.open(OUT, "w") as writer:
    writer.write_all(records)
print(f"Wrote {len(records)} records to {OUT}")
