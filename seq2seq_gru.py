"""Module implements RNN Seq2Seq model with attention mechanism."""
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


import matplotlib

from utils import get_text

matplotlib.rcParams.update({"figure.figsize": (16, 12), "font.size": 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output


def generate_translation(
    src: Tensor,
    src_len: Tensor,
    trg: Tensor,
    model: Any,
    trg_vocab: Any,
    src_vocab: Any,
):
    """Print source, target and translated sentences.

    Params:
        src: source sequence
        src_len: lengths without additions of each original sentence in the package
        trg: target sequence
        model: translator model
        trg_vocab: target vocabulary
        src_vocab: source vocabulary
    """
    model.eval()

    output = model(src, src_len, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    source = get_text(list(src[:, 0].cpu().numpy()), src_vocab)
    original = get_text(list(trg[:, 0].cpu().numpy()), trg_vocab)
    generated = get_text(list(output[1:, 0]), trg_vocab)

    print("Source: {}".format(" ".join(source)))
    print("Original: {}".format(" ".join(original)))
    print("Generated: {}".format(" ".join(generated)))
    print()


def train(
    model: Any,
    iterator: Any,
    optimizer: Any,
    criterion: Any,
    scheduler: Any,
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
        scheduler: scheduler for learning rate
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
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        output = output[1:].view(-1, output.shape[-1])
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

    scheduler.step()

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
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        n_layers: int,
        dropout: float,
        device: Any,
        max_length: int = 100,
    ):
        """Initialization of encoder part.

        Params:
            input_dim: source vocab length
            emb_dim: embedding dimension
            enc_hid_dim: hidden dimension of encoder RNN
            dec_hid_dim: hidden dimension of decoder RNN
            n_layers: number of RNN layers
            dropout: dropout probability
            device: cuda or gpu device
            max_length: maximum length of input sequence
        """
        super().__init__()

        self.device = device
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            num_layers=n_layers,
            bidirectional=True,
        )

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_len: Tensor) -> tuple[Tensor, Tensor]:
        """Return encoder hidden states and outputs.

        Params:
            src: source sequence
            src_len: lengths without additions of each original sentence in the package

        Returns:
            outputs, hidden
        """
        pos = (
            torch.arange(0, src.shape[0])
            .unsqueeze(0)
            .repeat(src.shape[1], 1)
            .to(self.device)
        )
        embedded = self.dropout(
            (self.embedding(src) * self.scale)
            + self.pos_embedding(pos).permute(1, 0, 2)
        )

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.to("cpu"), enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        attention: Any,
        device: Any,
        max_length: int = 100,
    ):
        """Initialization of decoder part.

        Params:
            output_dim: target vocab length
            emb_dim: embedding dimension
            enc_hid_dim: hidden dimension of encoder RNN
            dec_hid_dim: hidden dimension of decoder RNN
            dropout: dropout probability
            n_layers: number of RNN layers
            attention: attention module
            device: cuda or gpu device
            max_length: maximum length of input sequence
        """
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        self.rnn = nn.GRU(
            input_size=(enc_hid_dim * 2) + emb_dim, hidden_size=dec_hid_dim
        )

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: Tensor, hidden: Tensor, encoder_outputs: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return decoder outputs.

        Params:
            inputs: target sequence
            hidden: hidden states of encoder
            encoder_outputs: encoder outputs
            mask: mask to hide PAD tokens

        Return:
            prediction, hidden
        """
        inputs = inputs.unsqueeze(0)
        pos = (
            torch.arange(0, inputs.shape[0])
            .unsqueeze(0)
            .repeat(inputs.shape[1], 1)
            .to(self.device)
        )
        embedded = self.dropout(
            (self.embedding(inputs) * self.scale)
            + self.pos_embedding(pos).permute(1, 0, 2)
        )

        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, src_pad_idx: int, device: Any
    ):
        """Initialize Seq2Seq model (encoder, decoder).

        Params:
            encoder: encoder part
            decoder: decoder part
            src_pad_idx: index of PAD token in source vocabulary
            device: cuda or gpu

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src: Tensor) -> Tensor:
        """Returns mask for PAD tokens.

        Params:
            src: source sequence

        Return:
            mask for PAD tokens
        """
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(
        self,
        src: Tensor,
        src_len: Tensor,
        trg: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """Returns translation of input sequence

        Params:
            src: source sequence
            src_len: lengths without additions of each original sentence in the package
            trg: target sequence
            teacher_forcing_ratio: teacher forcing ratio

        Return:
            translation of input sequence
        """
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        trg_len = trg.shape[0]

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)

        inputs = trg[0, :]

        mask = self.create_mask(src)

        for t in range(1, trg_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs, mask)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            inputs = trg[t] if teacher_force else top1

        return outputs


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        """Initialization of attention.

        Params:
            enc_hid_dim: hidden dimension of encoder RNN
            dec_hid_dim: hidden dimension of decoder RNN
        """
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden: Tensor, encoder_outputs: Tensor, mask: Tensor) -> Tensor:
        """Returns attention scores.

        Params:
            hidden: hidden states of encoder
            encoder_outputs: encoder outputs
            mask: mask for PAD index

        Return:
            attention scores
        """
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)
