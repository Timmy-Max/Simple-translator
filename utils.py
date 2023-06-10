"""Module implements utility functions."""
from typing import Any
from torch import Tensor


def flatten(l: list) -> list:
    """Flattening of input.

    Params:
        l: list

    Returns:
        flattening of input
    """
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(
    sequence: list, tokens_to_remove: tuple = ("<eos>", "<sos>", "<unk>", "<pad>")
) -> list:
    """Remove tech tokens from sequence.

    Params:
        sequence: input sequence
        tokens_to_remove: tech tokens

    Return:
        sequence without tech tokens
    """
    return [x for x in sequence if x not in tokens_to_remove]


def get_text(indexes: list, trg_vocab: Any) -> list:
    """Return text sequence from word indexes.

    Params:
        indexes: word indexes
        trg_vocab: target vocabulary

    Return:
        text sequence
    """
    text = [trg_vocab.itos[token] for token in indexes]
    try:
        end_idx = text.index("<eos>")
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(
    src: Tensor,
    trg: Tensor,
    model: Any,
    trg_vocab: Any,
    src_vocab: Any,
    model_type: str = "default",
    src_len: Tensor = None,
):
    """Generates translation of source sequence.

    Params:
        src: source sequence
        trg: target sequence
        model: translator model
        trg_vocab: target vocabulary
        src_vocab: source vocabulary
        model_type: type of model
        src_len: lengths without additions of each original sentence in the package
    """
    model.eval()

    if model_type == "default":
        output = model(src, trg, 0)
    elif model_type == "src_len":
        output = model(src, src_len, trg, 0)

    output = output.argmax(dim=-1).cpu().numpy()

    source = get_text(list(src[:, 0].cpu().numpy()), src_vocab)
    original = get_text(list(trg[:, 0].cpu().numpy()), trg_vocab)
    generated = get_text(list(output[1:, 0]), trg_vocab)

    print("Source: {}".format(" ".join(source)))
    print("Original: {}".format(" ".join(original)))
    print("Generated: {}".format(" ".join(generated)))
    print()


def epoch_time(start_time: float, end_time: float) -> tuple[float, float]:
    """Computes training epoch time.

    Params:
        start_time: start time
        end_time: end_time

    Return:
        training epoch time
    """
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_sec
