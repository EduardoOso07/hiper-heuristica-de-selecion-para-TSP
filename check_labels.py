
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

def check_file(filename):
    path = ROOT / "results" / "data" / filename

    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = Counter(row["best_algorithm"] for row in rows)

    print(f"\nArchivo: {filename}")
    print(f"Total de instancias: {len(rows)}")
    print("-" * 40)

    for algorithm, count in labels.items():
        percentage = count / len(rows) * 100
        print(f"{algorithm}: {count} ({percentage:.2f}%)")

check_file("train_labeled.csv")
check_file("test_labeled.csv")
