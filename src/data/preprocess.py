from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterable

import pretty_midi
from tqdm import tqdm


def midi_to_events(midi_bytes: bytes, max_time_shift: int = 100) -> list[dict]:
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(("note_on", note.start, note.pitch, note.velocity))
            notes.append(("note_off", note.end, note.pitch, 0))

    notes.sort(key=lambda x: x[1])

    events = []
    current_time = 0.0
    time_scale = 100

    for event_type, event_time, pitch, velocity in notes:
        delta = int((event_time - current_time) * time_scale)

        while delta > 0:
            shift = min(delta, max_time_shift)
            events.append({"type": "time_shift", "value": shift})
            delta -= shift

        if event_type == "note_on":
            events.append({
                "type": "note_on",
                "pitch": int(pitch),
                "velocity": int(velocity),
            })
        else:
            events.append({
                "type": "note_off",
                "pitch": int(pitch),
            })

        current_time = event_time

    return events


def iter_midi_from_zip(zip_path: Path) -> Iterable[tuple[str, bytes]]:
    """
    Yield (filename, midi_bytes) from zip without full extraction.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.filename.endswith((".mid", ".midi")):
                with zf.open(member) as f:
                    yield member.filename, f.read()


def process_dataset(zip_path: Path,output_path: Path,limit: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, (filename, midi_bytes) in enumerate(
            tqdm(iter_midi_from_zip(zip_path), desc="Processing MIDI")
        ):
            if limit is not None and i >= limit:
                break

            try:
                events = midi_to_events(midi_bytes)
                record = {
                    "id": i,
                    "source": filename,
                    "events": events,
                }
                out_f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"[WARN] Failed on {filename}: {e}")


if __name__ == "__main__":
    zip_path = Path("../../midimizer_data/data/raw/maestro-v3.0.0.zip")
    output_path = Path("data/processed/maestro_events.jsonl")

    process_dataset(zip_path=zip_path, output_path=output_path)