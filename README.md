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

This project implements a deep learning pipeline for generating MIDI music using sequence models such as LSTM, GRU and Transformer.

The model is trained on a MIDI dataset (MAESTRO) and learns to generate music by predicting the next musical event in a sequence.

The pipeline includes:
- MIDI preprocessing
- event-based representation
- tokenization
- sequence chunking
- model training
- MIDI generation
