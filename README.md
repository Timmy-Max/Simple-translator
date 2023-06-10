# Simple_translator
In one of the homework assignments for the machine learning course, there was a task to make a translator with BLEU more than 27. This repository shows attempts to do this.
# Data
A parallel Russian-English corpus of texts is used for teaching. The theme of the texts is travel and tourism. The total volume is 50,000.
# Models
## Seq2Seq LSTM
Simple three layers encoder-decoder network with LSTM cells.
![image](https://github.com/Timmy-Max/Simple_translator/assets/51882110/67ebb67b-99b5-4469-a5e6-3951e6fe7750)
## Seq2seq GRU with attention
The encoder consists of a bidirectional GRU (two layers). The encoder consists of a unidirectional GRU (single layer). The mechanism of attention is applied.
![image](https://github.com/Timmy-Max/Simple_translator/assets/51882110/1b8b7dfa-d2d3-4065-b41e-3cd0eacfcb98)
## Transformer
Developed by: Language Technology Research Group at the University of Helsinki; model type: transformer-align.
# Evaluation
|   | Model                     | Test Loss   | Test BLEU | Parametrs | Train Time |
|---|---------------------------|-------------|---------- |-----------|------------|
| 1 | Simple RNN Seq2Seq     | 4.542       | 13.299    | ~ 23 m    | ~ 40 min   |
| 2 | Attention Seq2seq      | 4.107       | 29.745    | ~ 36 m    | ~ 120 min  |
| 3 | Fine-tuned transformer | 1.007       | 36.984    | ~ 77 m    | ~ 30 min   |
