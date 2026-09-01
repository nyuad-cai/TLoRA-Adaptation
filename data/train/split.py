import json
import random
import argparse


def split_dataset(input_file, train_out, val_out, val_ratio=0.1, seed=42):
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples")

    print(f"Loaded {len(data)} samples")

    # Reproducible shuffle
    random.seed(seed)
    random.shuffle(data)

    val_size = int(len(data) * val_ratio)

    val_data = data[:val_size]
    train_data = data[val_size:]

    print(f"Validation size: {len(val_data)}")
    print(f"Train size: {len(train_data)}")

    # Save files
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print("Split complete.")
    print(f"Train saved to: {train_out}")
    print(f"Validation saved to: {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--train_out", type=str, default="train_new.json")
    parser.add_argument("--val_out", type=str, default="val.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_dataset(
        args.input_file,
        args.train_out,
        args.val_out,
        args.val_ratio,
        args.seed
    )