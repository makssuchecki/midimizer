from pathlib import Path
import json
from tqdm import tqdm


def event_to_token(event: dict) -> str:
    if event["type"] == "time_shift":
        return f"time_shift_{event['value']}"
    elif event["type"] == "note_on":
        return f"note_on_{event['pitch']}_{event['velocity']}"
    elif event["type"] == "note_off":
        return f"note_off_{event['pitch']}"
    else:
        raise ValueError(f"Unknown event {event}")


def build_vocab(input_path: Path, vocab_path: Path):
    vocab = set()

    with input_path.open() as f:
        for line in tqdm(f, desc="Building vocab"):
            record = json.loads(line)
            for event in record["events"]:
                token = event_to_token(event)
                vocab.add(token)

    vocab = sorted(vocab)
    token_to_id = {tok: i for i, tok in enumerate(vocab)}

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w") as f:
        json.dump(token_to_id, f, indent=2)

    print(f"Vocab size: {len(token_to_id)}")


def tokenize_dataset(input_path: Path, vocab_path: Path, output_path: Path):
    with vocab_path.open() as f:
        token_to_id = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in tqdm(fin, desc="Tokenizing"):
            record = json.loads(line)

            tokens = [
                token_to_id[event_to_token(e)]
                for e in record["events"]
            ]

            out = {
                "id": record["id"],
                "tokens": tokens
            }

            fout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    input_path = Path("data/processed/maestro_events.jsonl")
    vocab_path = Path("data/index/vocab.json")
    output_path = Path("data/processed/maestro_tokens.jsonl")

    build_vocab(input_path, vocab_path)
    tokenize_dataset(input_path, vocab_path, output_path)