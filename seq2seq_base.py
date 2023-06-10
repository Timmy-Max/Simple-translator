"""Module implements basic RNN Seq2Seq translator."""
import random
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

import matplotlib

matplotlib.rcParams.update({"figure.figsize": (16, 12), "font.size": 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output


def train(
    model: Any,
    iterator: Any,
    optimizer: Any,
    criterion: Any,
    clip: float,
    train_history: list = None,
    valid_history: list = None,
) -> float:
    """Function implements one epoch training for model.

    Params:
        model: model for training
        iterator: train data iterator
        optimizer: model optimizer
        criterion: loss function
        clip: limiting the gradient norm
        train_history: train loss after each previous epoch
        valid_history: validation loss after each previous epoch

    Returns:
        float: train loss after this epoch
    """
    model.train()
    epoch_loss = 0
    history = []

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())

        if (i + 1) % 10 == 0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label="train loss")
            ax[0].set_xlabel("Batch")
            ax[0].set_title("Train loss")
            if train_history is not None:
                ax[1].plot(train_history, label="general train history")
                ax[1].set_xlabel("Epoch")
            if valid_history is not None:
                ax[1].plot(valid_history, label="general valid history")
            plt.legend()

            plt.show()

    return epoch_loss / len(iterator)


def evaluate(model: Any, iterator: Any, criterion: Any) -> float:
    """Function implements evaluation of the model.

    Params:
        model: model for evaluation
        iterator: validation data iterator
        criterion: loss function

    Returns:
        float: validation loss
    """
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ):
        """Initialization of encoder part.

        Params:
            input_dim: source vocab length
            emb_dim: embedding dimension
            hid_dim: hidden dimension of RNN
            n_layers: number of RNN layers
            dropout: dropout probability
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """Return encoder hidden and cell states.

        Params:
            src: source sequence

        Returns:
            hidden, cell
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ):
        """Initialization of decoder part.

        Params:
            output_dim: target vocab length
            emb_dim: embedding dimension
            hid_dim: hidden dimension of RNN
            n_layers: number of RNN layers
            dropout: dropout probability

        """
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, cell) -> tuple[Tensor, Tensor, Tensor]:
        """Return decoder outputs.

        Params:
            inputs: target sequence
            hidden: hidden states of encoder
            cell: cell states of encoder

        Return:
            prediction, hidden, cell
        """
        inputs = inputs.unsqueeze(0)
        embedded = self.dropout(self.embedding(inputs))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: Any):
        """Initialize Seq2Seq model (encoder, decoder).

        Params:
            encoder: encoder part
            decoder: decoder part
            device: cuda or gpu

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        """Returns translation of input sequence

        Params:
            src: source sequence
            trg: target sequence
            teacher_forcing_ratio: teacher forcing ratio

        Return:
            translation of input sequence
        """
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        inputs = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inputs = trg[t] if teacher_force else top1

        return outputs
