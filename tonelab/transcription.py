"""
transcription.py
Automatic Tone Transcription Module
"""
import torch

def normalization_label(Y, use_sigmoid = True):
    """
    Normalize the labels in the range [0, 1]

    Args:
      Y: torch.tensor[num_samples, 3]

    Returns:
      normalized_Y: torch.tensor[num_samples, 3
    """
    normalize_Y = torch.zeros((Y.shape[0], Y.shape[1]))

    for i in range(Y.shape[0]):
        if Y[i][-1] == 0:
          normalize_Y[i][0], normalize_Y[i][1], normalize_Y[i,2] = Y[i][0], 0.5*Y[i][0] + 0.5*Y[i][1], Y[i][1]
        else:
          normalize_Y[i][0], normalize_Y[i][1], normalize_Y[i,2] = Y[i][0], Y[i][1], Y[i][2]

        min_value = normalize_Y[i].min()
        max_value = normalize_Y[i].max()
        if min_value != max_value:
            normalize_Y[i] = (normalize_Y[i] - min_value) / (max_value - min_value)
        else:
            normalize_Y[i][0], normalize_Y[i][1], normalize_Y[i,2] = 1, 1, 1

    if use_sigmoid:
        normalize_Y = torch.sigmoid(normalize_Y)

    return normalize_Y


def all_normalization_seq(use_sigmoid=True):
    """
    Generate and normalize all potential sequences, and identify unique sequences.

    Returns:
        index_list: List of lists, where each sublist contains the original sequences
                    that correspond to a unique normalized sequence.
        unique_rows: torch.tensor, unique normalized sequences.
    """
    num_sequences = 150
    all_seq = torch.zeros((num_sequences, 3))
    all_seq_str = []

    flag = 0
    for i in range(1, 6):
        for j in range(1, 6):
            all_seq_str.append([i, j])
            all_seq[flag] = torch.tensor([i, 0.5 * i + 0.5 * j, j])
            flag += 1
            for k in range(1, 6):
                all_seq_str.append([i, j, k])
                all_seq[flag] = torch.tensor([i, j, k])
                flag += 1

    # Normalize all sequences
    normalize_all_seq = normalization_label(Y=all_seq, use_sigmoid=False)

    # Find unique rows and their indices
    unique_rows, inverse_indices = torch.unique(normalize_all_seq, dim=0, return_inverse=True)
    index_list = [[] for _ in range(len(unique_rows))]

    for idx, unique_idx in enumerate(inverse_indices):
        index_list[unique_idx].append(all_seq_str[idx])

    return index_list, unique_rows

def match_transcription(feat, unique_transcription):
    """
    Matches the given feature vector to the closest row in the unique transcription tensor.

    Args:
    - feat: torch.tensor of shape (3), the feature vector to match.
    - unique_transcription: torch.tensor of shape (37, 3), the tensor containing unique transcriptions.

    Returns:
    - torch.tensor of shape (3), the closest matching row from unique_transcription.
    """

    # Compute the absolute difference between feat and each row in unique_transcription
    diff = torch.abs(unique_transcription - feat)

    # Sum the absolute differences along the second dimension to get a single value for each row
    diff_sum = diff.sum(dim=1)

    # Find the index of the row with the smallest sum of differences
    min_index = torch.argmin(diff_sum)

    # Return the row from unique_transcription that has the smallest difference
    return unique_transcription[min_index]
