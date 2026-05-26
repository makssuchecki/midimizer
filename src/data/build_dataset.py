import json
from pathlib import Path
from tqdm import tqdm

'''
Dzielenie długich sekwencji na kawałki
'''
def chunk_tokens(tokens, seq_len=256):
    for i in range(0, len(tokens) - seq_len, seq_len):
        yield tokens[i:i+seq_len]


def build_chunks(input_path: Path, output_path: Path, seq_len=256):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in tqdm(fin, desc="Chunking"):
            record = json.loads(line)
            tokens = record["tokens"]

            for chunk in chunk_tokens(tokens, seq_len):
                fout.write(json.dumps({"tokens": chunk}) + "\n")


if __name__ == "__main__":
    build_chunks(
        Path("data/processed/maestro_tokens.jsonl"),
        Path("data/processed/train_chunks.jsonl"),
        seq_len=256
    )