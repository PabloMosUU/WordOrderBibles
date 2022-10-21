import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np


def unigram_entropy(model: nn.Module) -> float:
    # Todo: challenge 4a
    raise NotImplementedError()


def entropy_rate(model: nn.Module, encodings: torch.Tensor, stride: int, device: str) -> float:
    """
    Compute the entropy rate (entropy per word) for a given text, given a model and a stride for the sliding window
    :param model: the language model that is used to compute probabilities
    :param encodings: the text for which you want to estimate the entropy, encoded as token IDs
    :param stride: the stride for the sliding window approach. 1 is call LM once per token
    :param device: the device where you want to run the computations (cuda or cpu)
    :return: an entropy rate (entropy per word) in bits per word
    """
    max_length = model.config.n_positions
    seq_len = encodings.size(1)

    neg_log_likelihoods = []
    end_loc, prev_end_loc = 0, 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # CrossEntropyLoss averages over input tokens. Multiply it with trg_len to get the summation instead.
            # We will take average over all the tokens in the last step.
            neg_log_likelihood = outputs.loss * trg_len

        neg_log_likelihoods.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Division by np.log(2) is change of base to base-2 logarithm
    # return torch.stack(losses).sum() / end_loc / np.log(2)
    return torch.stack(neg_log_likelihoods).sum() / end_loc / np.log(2)


def entropy_rate_pablo(model: nn.Module, encodings: torch.Tensor, stride: int, device: str) -> float:
    assert len(encodings) == 1
    encodings = encodings[0]
    model.eval()
    max_length = model.config.n_positions
    begin_loc, end_loc = 0, max_length
    seq_len = len(encodings)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    losses = []
    with torch.no_grad():
        while begin_loc < seq_len:
            # We want to avoid considering the tokens that we have already considered before
            first_logit = 1 if begin_loc == 0 else end_loc - 1 - begin_loc  # TODO: 1 is a magic number coming from 2 prompt tokens
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids = encodings[begin_loc:end_loc].to(device)
            logits = model(input_ids).logits
            loss = loss_fn(logits[first_logit:-1], input_ids[first_logit + 1:]).item()
            losses.append(loss)
            begin_loc += stride
    total_loss = sum(losses)
    return (total_loss / (seq_len - 2)) / np.log(2)


def entropy_rate_no_stride(model: torch.nn.Module, encodings: torch.Tensor):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        assert len(encodings) == 1
        outputs = model(encodings[0])
        n_input_tokens = len(encodings[0])
        fn_loss = loss_fn(outputs.logits[1:n_input_tokens - 1], encodings[0][2:])
    float_loss = fn_loss.item()
    return float_loss / np.log(2)


def entropy_rate_no_stride_old(model: nn.Module, encodings: torch.Tensor):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        assert len(encodings) == 1
        seq = encodings[0]
        outputs = model(seq)
        n_prompt_tokens = 2  # TODO: magic variable
        n_input_tokens = len(seq)
        fn_loss = loss_fn(outputs.logits[n_prompt_tokens - 1:n_input_tokens - 1], seq[n_prompt_tokens:])
    return fn_loss.item() / np.log(2)
