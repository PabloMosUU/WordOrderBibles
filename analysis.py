import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np


def unigram_entropy(model: nn.Module) -> float:
    # Todo: challenge 4a
    raise NotImplementedError()


def entropy_rate(model: nn.Module, encodings: torch.Tensor, stride: int, device: str, mask_prompt_tokens: int) -> float:
    """
    Compute the entropy rate (entropy per word) for a given text, given a model and a stride for the sliding window
    :param model: the language model that is used to compute probabilities
    :param encodings: the text for which you want to estimate the entropy, encoded as token IDs
    :param stride: the stride for the sliding window approach. 1 is call LM once per token
    :param device: the device where you want to run the computations (cuda or cpu)
    :param mask_prompt_tokens: number of tokens to mask at the beginning of the entire text (prompt)
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

        # Mask prompt tokens
        if begin_loc == 0 and mask_prompt_tokens > 0:
            target_ids[:, :mask_prompt_tokens] = -100

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
