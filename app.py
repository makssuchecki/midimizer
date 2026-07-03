import json
import tempfile
from pathlib import Path

import gradio as gr
import torch

from src.model.lstm import LSTMModel
from src.model.gru import GRUModel
from src.model.transformer import TransformerModel
from src.utils.midi_writer import events_to_midi

VOCAB_PATH = Path("data/index/vocab.json")
CHECKPOINTS_DIR = Path("outputs/trained")

MODEL_CLASSES = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "Transformer": TransformerModel,
}

CHECKPOINT_PREFIX = {
    "LSTM": "lstm",
    "GRU": "gru",
    "Transformer": "tra",
}


def load_vocab():
    with VOCAB_PATH.open() as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    return vocab, id_to_token


def list_checkpoints(arch: str = None):
    if not CHECKPOINTS_DIR.exists():
        return []
    prefix = CHECKPOINT_PREFIX.get(arch) if arch else None
    return sorted(
        str(p) for p in CHECKPOINTS_DIR.rglob("*.pt")
        if prefix is None or p.stem.startswith(prefix)
    )


def sample(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def tokens_to_events(tokens, id_to_token):
    events = []
    for t in tokens:
        tok = id_to_token[t]
        if tok.startswith("time_shift"):
            events.append({"type": "time_shift", "value": int(tok.split("_")[2])})
        elif tok.startswith("note_on"):
            _, _, pitch, vel = tok.split("_")
            events.append({"type": "note_on", "pitch": int(pitch), "velocity": int(vel)})
        elif tok.startswith("note_off"):
            _, _, pitch = tok.split("_")
            events.append({"type": "note_off", "pitch": int(pitch)})
    return events


def generate_midi(model_arch: str, checkpoint_path: str, length: int, temperature: float):
    if not checkpoint_path:
        raise gr.Error("Select a checkpoint first.")

    vocab, id_to_token = load_vocab()
    vocab_size = len(vocab)

    model_cls = MODEL_CLASSES[model_arch]
    model = model_cls(vocab_size)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tokens = [0]
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor(tokens).unsqueeze(0)
            logits = model(x)
            next_token = sample(logits[0, -1], temperature)
            tokens.append(next_token)

    events = tokens_to_events(tokens, id_to_token)

    out_file = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    events_to_midi(events, out_file.name)

    return out_file.name


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#1c1c1c",
    body_background_fill_dark="#1c1c1c",
    block_background_fill="#2a2a2a",
    block_background_fill_dark="#2a2a2a",
    block_border_color="#3d3d3d",
    block_border_color_dark="#3d3d3d",
    input_background_fill="#333333",
    input_background_fill_dark="#333333",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_label_text_color="#e0e0e0",
    block_label_text_color_dark="#e0e0e0",
)


def build_app():
    initial_arch = "GRU"
    checkpoints = list_checkpoints(initial_arch)

    with gr.Blocks(title="Midimizer", theme=THEME) as demo:
        gr.Markdown("# Midimizer\nGenerate MIDI using a trained sequence model.")

        with gr.Row():
            with gr.Column():
                model_arch = gr.Dropdown(
                    choices=list(MODEL_CLASSES.keys()),
                    value=initial_arch,
                    label="Model architecture",
                )
                checkpoint = gr.Dropdown(
                    choices=checkpoints,
                    value=checkpoints[0] if checkpoints else None,
                    label="Checkpoint",
                )
                refresh_btn = gr.Button("Refresh checkpoints", size="sm")

                length = gr.Slider(
                    minimum=50, maximum=2000, value=300, step=50, label="Generation length (tokens)"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="Temperature"
                )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                midi_output = gr.File(label="Download generated MIDI")

        def update_checkpoints(arch):
            updated = list_checkpoints(arch)
            return gr.Dropdown(choices=updated, value=updated[0] if updated else None)

        model_arch.change(fn=update_checkpoints, inputs=model_arch, outputs=checkpoint)
        refresh_btn.click(fn=update_checkpoints, inputs=model_arch, outputs=checkpoint)

        generate_btn.click(
            fn=generate_midi,
            inputs=[model_arch, checkpoint, length, temperature],
            outputs=midi_output,
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
