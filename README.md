# Simple_translator
In one of the homework on machine learning course, there was a task to make a translator with BLEU more than 27. This repository shows attempts to do this.
# Data
A parallel Russian-English corpus of texts is used for teaching. The theme of the texts is travel and tourism. The total volume is 50,000.
# Evaluation
|   | Model                     | Test Loss   | Test BLEU | Parametrs | Train Time |
|---|---------------------------|-------------|---------- |-----------|------------|
| 1 | Simple RNN Seq2Seq     | 4.542       | 13.299    | ~ 23 m    | ~ 40 min   |
| 2 | Attention Seq2seq      | 4.107       | 29.745    | ~ 36 m    | ~ 120 min  |
| 3 | Fine-tuned transformer | 1.007       | 36.984    | ~ 77 m    | ~ 30 min   |
