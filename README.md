---
title: Midimizer
emoji: 🎹
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# Midimizer

A deep learning pipeline for generating piano music in MIDI format. Sequence models (LSTM, GRU, Transformer) are trained on the MAESTRO dataset to predict musical events one step at a time — the same idea as a language model, but for music.


## How it works

MIDI files are converted into a flat sequence of discrete events:

These tokens are mapped to integer IDs and the model is trained to predict the next token given all previous ones (next-token prediction / language modelling). At generation time, tokens are sampled autoregressively and decoded back into a `.mid` file.


## Models

| Architecture | Embedding | Hidden | Layers |
|---|---|---|---|
| LSTM | 256 | 512 | 2 |
| GRU | 128 | 256 | 1 |
| Transformer | 128 | — | 2 (4 heads) |


## Setup

```bash
git clone https://github.com/makssuchecki/midimizer.git
cd midimizer
pip install -r requirements.txt
pip install torch
```

Download the MAESTRO v3 dataset and place the zip at `../midimizer_data/data/raw/maestro-v3.0.0.zip`.

---

## Usage

**1. Preprocess**
```bash
python -m src.data.preprocess
```

**2. Tokenize**
```bash
python -m src.data.tokenizer
```

**3. Build training chunks**
```bash
python -m src.data.build_dataset
```

**4. Train**
```bash
python -m src.train
```

**5. Generate**
```bash
python -m src.generate
```

**6. Run the GUI**
```bash
python app.py
```
Opens a Gradio interface in the browser where you can pick a model, set generation length and temperature, and download the generated `.mid` file.

---

## Project structure

```
src/
├── data/
│   ├── preprocess.py      # MIDI to event sequences
│   ├── tokenizer.py       # build vocab, events
│   ├── build_dataset.py   # chunk sequences for training
│   └── dataset.py         # PyTorch Dataset
├── model/
│   ├── lstm.py
│   ├── gru.py
│   └── transformer.py
├── utils/
│   └── midi_writer.py     # events to .mid file
├── train.py
└── generate.py
app.py                     # Gradio GUI
```
