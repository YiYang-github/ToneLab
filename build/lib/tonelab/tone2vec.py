"""
tone2vec.py
Tone2Vec Module: Map Transcriptions into Representations
"""

import re
import os
import umap
import numpy as np
import pandas as pd
from typing import List, Callable
from collections import Counter
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def pitch_curve(seq: List[float]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a simulated smooth pitch variation curve based on the given tone transcription sequence.

    The pitch variation curve depends on the length of the sequence:

    - For sequences of length 2, a straight line connecting the points (1, x1) and (3, x2) is generated.
    - For sequences of length 3, a quadratic curve is fitted to represent the smooth variation of pitch.

    Args:
        seq (List[float]): A list of pitch levels represented by numbers in the range [1, 5].

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that, when provided an array of x values, returns the corresponding y values representing the pitch curve.
    """

    if len(seq) == 2:
        # For a sequence of length 2, use a straight line connecting (1, x1) and (3, x2)
        coeffs = np.polyfit([1, 3], seq, 1)
        return np.poly1d(coeffs)
    elif len(seq) == 3:
        # For a sequence of length 3, fit a quadratic curve to represent pitch variations
        coeffs = np.polyfit([1, 2, 3], seq, 2)
        return np.poly1d(coeffs)
    else:
        raise ValueError("Sequence length must be 2 or 3.")



def curve_simi(seq1: List[float], seq2: List[float]) -> float:
    """
    Calculates the area of the absolute difference between two pitch curves over the domain [1, 3].

    Args:
        seq1 (List[float]): The first sequence of pitch levels.
        seq2 (List[float]): The second sequence of pitch levels.

    Returns:
        float: The area representing the difference between the two pitch curves.
    """

    # Calculate the pitch curves for both sequences
    curve1 = pitch_curve(seq1)
    curve2 = pitch_curve(seq2)
    diff_curve = lambda x: np.abs(curve1(x) - curve2(x))

    # Use numerical integration to estimate the area using the trapezoidal rule
    x = np.linspace(1, 3, 1000)  # Generate 1000 points between 1 and 3
    y = diff_curve(x)
    area = np.trapz(y, x)  # Numerical integration using the trapezoidal rule
    return area


def cal_simi(save_dir: str) -> np.ndarray:
    """
    Calculate the similarity between sequences of pitch levels.

    This function iterates through all possible combinations of sequences,
    computes their similarity using the `curve_simi` function, and stores
    the results in a 6-dimensional tensor.

    Args:
        save_dir (str): The directory where the similarity results will be saved.

    Returns:
        np.ndarray: A 6D array containing the similarity values for the sequences.
    """
    
    # Initialize a 6D array to hold similarity values
    similarity_tensor = np.zeros((6, 6, 6, 6, 6, 6))

    # Iterate through all possible combinations of sequences
    for i1 in range(1, 6):
        for i2 in range(1, 6):
            for i3 in range(1, 6):
                for j1 in range(1, 6):
                    for j2 in range(1, 6):
                        for j3 in range(1, 6):
                            seq1 = [i1, i2, i3]
                            seq2 = [j1, j2, j3]

                            # Remove trailing zeros, if any
                            seq1 = [x for x in seq1 if x != 0]
                            seq2 = [x for x in seq2 if x != 0]

                            # Calculate similarity if both sequences are non-empty
                            if seq1 and seq2:
                                similarity = curve_simi(seq1, seq2)

                                # Fill the tensor, considering symmetry
                                similarity_tensor[i1][i2][i3][j1][j2][j3] = similarity
                                similarity_tensor[j1][j2][j3][i1][i2][i3] = similarity

    # Save the results to the specified directory if it exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the path for saving the similarity tensor
    np.save(os.path.join(save_dir, 'similarity_tensor.npy'), similarity_tensor)

    return similarity_tensor



def Levenshtein_distance(list1, list2, transition_matrix=False):
    """
    A General Extension to Levenshtein Distance (Edit Distance). 
    
    See Wieling, M., Margaretha, E., & Nerbonne, J. (2012). Inducing a measure of phonetic similarity from pronunciation variation. J. Phonetics, 40, 307-314.

    Args:
        transition_matrix: np.array [n+1, n+1]; (i, j) represents the cost transferring i to j.
                           The transition_matrix for edit distance is np.ones((n+1, n+1)) - np.identity(n+1)

        list1, list2: the elements are from 1 to n (1-based indexing)

    Output:
        returns the segment x segment frequency matrix and the overall cost
    """

    transition_matrix = np.ones((100, 100)) - np.identity(100) if not transition_matrix else transition_matrix

    len1, len2 = len(list1), len(list2)
    dp = np.zeros((len1 + 1, len2 + 1))
    count_joint_p = np.zeros((transition_matrix.shape[0], transition_matrix.shape[0]))

    # Initialize dp array for deletions and insertions
    for i in range(1, len1 + 1):
        dp[i][0] = dp[i-1][0] + transition_matrix[list1[i-1], 0] 

    for j in range(1, len2 + 1):
        dp[0][j] = dp[0][j-1] + transition_matrix[0, list2[j-1]]
    
    # Dynamic programming to solve subproblems
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            replace_cost = dp[i-1][j-1] + transition_matrix[list1[i-1], list2[j-1]]
            insert_cost = dp[i][j-1] + transition_matrix[0, list2[j-1]]
            delete_cost = dp[i-1][j] + transition_matrix[list1[i-1], 0]

            # Choose the minimum cost
            dp[i][j] = min(replace_cost, insert_cost, delete_cost)

            # Update the count_joint_p based on the chosen operation
            if dp[i][j] == replace_cost:
                count_joint_p[list1[i-1], list2[j-1]] += 1
            elif dp[i][j] == insert_cost:
                count_joint_p[0, list2[j-1]] += 1
            elif dp[i][j] == delete_cost:
                count_joint_p[list1[i-1], 0] += 1
    
    return dp[len1][len2], count_joint_p


def loading(file_path: str, column_name: str = None) -> list:
    """
    Read data from a CSV or XLSX file. If a column name is specified, return that column as a list.
    Otherwise, return the data as a list of lists, skipping the first column of each row.

    Parameters:
        file_path (str): The path to the file including the filename and extension.
        column_name (str, optional): Name of the column to extract. If None, skip the first column of each row.

    Returns:
        list: Depending on the input, returns a single column or all data with the first column skipped.
    """
    # Determine the file type and read the file accordingly
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use '.xlsx' or '.csv'.")

    # Check if a specific column name has been provided
    if column_name:
        if column_name in df.columns:
            return df[column_name].tolist()
        else:
            raise ValueError(f"Column name '{column_name}' not found in the file.")
    else:
        # Return all data except the first column in list of lists format
        return df.iloc[:, 1:].values.tolist()

def parse_phonemes(data: List[List[str]]) -> Tuple[List[List[List[int]]], List[List[List[int]]], List[List[List[int]]], List[List[List[int]]]]:
    """
    Parses lists of speech strings into structured phoneme data.

    Args:
        data (List[List[str]]): Each sublist represents dialect speeches in the format 't ɔ 55'.

    Returns:
        Tuple of lists for initials, finals, tones, and combined phoneme data, each structured as nested lists.
    """

    phoneme_counter = Counter()
    special_chars = [ '-', '\x8d', '）', '无', '？', '\uf179', '͊', ')', '（', '̥', '̱', '\x8d', '̍', '\uf179', ')']


    for dialect_list in data:
        for phoneme_str in dialect_list:

            for char in special_chars:
                phoneme_str = phoneme_str.replace(char, '')  
            raw_phonemes = phoneme_str.split()[:-1]  

            for phoneme in raw_phonemes:
                for ele in phoneme:
                    phoneme_counter[ele] += 1

    unique_phonemes_list = sorted(phoneme_counter.items(), key=lambda x: x[1])

    unique_phonemes = []

    for phoneme, count in phoneme_counter.items():
        if count > 10: 
            unique_phonemes.append(phoneme)

    print(f"There are {len(unique_phonemes)} unique phonemes")
    print("Filtered phonemes:", unique_phonemes)
    phoneme2num = {}
    for index in range(len(unique_phonemes)):
        phoneme2num[unique_phonemes[index]] = index + 1

    
    initials_lists, finals_lists, tones_lists, all_lists = [], [], [], []
    for dialect in data:
        initials, finals, alls, tones = [], [], [], []
        for speech in dialect:
            if validate_speech(speech, phoneme2num, special_chars):
                all_phonemes, initial_phonemes, final_phonemes, tone_phonemes = validate_speech(speech, phoneme2num, special_chars)
                initials.append(initial_phonemes)
                finals.append(final_phonemes)
                alls.append(all_phonemes)
                tones.append(tone_phonemes)
            else:
                initials.append([])
                finals.append([])
                alls.append([])
                tones.append([])

        initials_lists.append(initials)
        finals_lists.append(finals)
        all_lists.append(alls)
        tones_lists.append(tones)

    return initials_lists, finals_lists, all_lists, tones_lists


def validate_speech(speech: str, phoneme_to_num: Dict[str, int], special_chars: List[str]) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
    """
    Validates the format of a speech string and returns its components if valid.

    Args:
        speech (str): Speech string to be validated.
        phoneme_to_num (Dict[str, int]): Mapping of phonemes to numbers.
        special_chars (List[str]): Characters that invalidate the speech.

    Returns:
        A tuple of lists for all phonemes, initials, finals, and tones if valid, otherwise None.
    """
    if any(char in speech for char in special_chars) or len(speech.split()) != 3:
        return None

    parts = speech.split()
    initials_part, finals_part, tone_string = parts
    initials = [phoneme_to_num[phon] for phon in initials_part if phon in phoneme_to_num]
    finals = [phoneme_to_num[phon] for phon in finals_part if phon in phoneme_to_num]

    if not initials or not finals or not re.fullmatch(r'[1-5]{1,3}', tone_string):
        return None

    tones = [int(tone) for tone in tone_string]
    all_phonemes = initials + finals

    return all_phonemes, initials, finals, tones



def calculate_sparsity(phoneme_lists: List[List[List[int]]]):
    """
    Calculates and prints the sparsity of the phoneme data.

    Args:
        phoneme_lists (List[List[List[int]]]): Nested list of phoneme data.
    """
    total = sum(len(sublist) for phoneme_list in phoneme_lists for sublist in phoneme_list)
    non_empty = sum(bool(item) for phoneme_list in phoneme_lists for sublist in phoneme_list for item in sublist)
    sparsity = non_empty / total if total else 0
    print(f"The data sparsity is {sparsity:.6f}.")


def fill_with_mode(tone_list):
    """
    Fill empty lists in tone_list with the most common non-empty list of their respective columns.
    Removes columns that are entirely empty after this processing.

    Parameters:
        tone_list (list of list of list): A 2D list where each element is a list that might be empty.

    Returns:
        list: The tone_list with empty lists filled with the most common non-empty list of their respective columns,
              and columns that remain entirely empty are removed.
    """
    num_rows = len(tone_list)
    num_cols = len(tone_list[0]) if num_rows > 0 else 0

    # Get the modes for each column
    column_modes = []
    for col in range(num_cols):
        # Collect all non-empty lists in the column
        column_data = [tuple(tone_list[row][col]) for row in range(num_rows) if tone_list[row][col]]
        # Calculate mode using Counter, directly on tuples for simplicity
        if column_data:
            most_common_structure = Counter(column_data).most_common(1)[0][0]
        else:
            most_common_structure = ()
        column_modes.append(list(most_common_structure))  # Convert tuple back to list

    # Fill empty lists with the mode of their respective column
    for row in range(num_rows):
        for col in range(num_cols):
            if not tone_list[row][col]:
                if column_modes[col]:  # Only fill if there is a mode to fill with
                    tone_list[row][col] = column_modes[col].copy()  # Copy to prevent aliasing

    # Remove columns that are still entirely empty
    columns_to_remove = [col for col in range(num_cols) if not any(tone_list[row][col] for row in range(num_rows))]
    for row in range(num_rows):
        for col in sorted(columns_to_remove, reverse=True):  # Reverse to avoid shifting indices
            del tone_list[row][col]

    return tone_list



def tone_feats(tone_list, method='PCA', dim=2):
    """
    Apply PCA, t-SNE, or UMAP to reduce the dimensionality of the data and extract tone features based on indices.

    Parameters:
        tone_list (list of list of tuples): Indices in the format [(x, y, z), ...] for each dialect and word.
        method (str): The dimensionality reduction method to use ('PCA', 't-SNE', 'UMAP').
        dim (int): The target dimensionality for reduction.

    Returns:
        np.ndarray: The extracted tone features reshaped based on the input indices.
    """
    # Load and reshape the data
    data_tone2vec = np.load('tonelab/weights/tone2vec.npy')
    reshaped_data = data_tone2vec.reshape(-1, 216)  # Assuming data_tone2vec is originally (6, 6, 6, 6, 6, 6)

    # Choose and initialize the dimensionality reduction model
    if method == 'PCA':
        model = PCA(n_components=dim)
    elif method == 't-SNE':
        model = TSNE(n_components=dim, learning_rate='auto', init='random')
    elif method == 'UMAP':
        model = umap.UMAP(n_components=dim)
    else:
        raise ValueError("Unsupported dimensionality reduction method specified.")

    # Fit and transform the data
    reduced_data = model.fit_transform(reshaped_data)
    reduced_data = reduced_data.reshape(6, 6, 6, dim)  # Reshaping back assuming the original shape

    tone_list = fill_with_mode(tone_list)
    num_dialect, num_word = len(tone_list), len(tone_list[0])
    tone_feats = np.zeros((num_dialect, num_word, dim))

    # Populate the tone_feats array with reduced data based on indices from tone_list
    for i in range(num_dialect):
        for j in range(num_word):
            indices = tone_list[i][j]
            if len(indices) == 3:
                x, y, z = indices
            elif len(indices) == 2:
                x, y, z = indices[0], indices[1], 0  # Handle cases where z index is missing
            else:
                x, y, z = indices[0], 0, 0  # Handle cases where z index is missing
                

            # Assigning the reduced features to the tone_feats array
            tone_feats[i, j] = reduced_data[x, y, z]

    tone_feats = tone_feats.reshape(num_dialect, num_word * dim)

    return tone_feats


def plot(feats, labels, method='PCA'):
    """
    Reduces the dimensionality of the feature array to 2D and visualizes the results,
    coloring the points according to the provided labels.

    Parameters:
        feats (np.ndarray): An array of shape (n, m) with n samples and m features.
        labels (list): A list of length n with string labels for each sample.
        method (str): The method used for dimensionality reduction ('PCA', 't-SNE', 'UMAP').

    Returns:
        None: This function visualizes the 2D projection of the data.
    """
    # Convert labels to integers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(np.unique(labels_encoded))
    print(f"Number of unique labels (k): {num_classes}")

    # Initialize the dimensionality reduction model
    if method == 'PCA':
        model = PCA(n_components=2)
    elif method == 't-SNE':
        model = TSNE(n_components=2)
    elif method == 'UMAP':
        model = umap.UMAP(n_components=2)
    else:
        raise ValueError("Unsupported dimensionality reduction method specified.")

    # Reduce dimensions
    reduced_data = model.fit_transform(feats)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar(scatter, ticks=range(num_classes), label='Label classes')
    plt.title(f'{method} Reduction of Features to 2D')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

def phoneme_feats(all_list):
    """
    Computes a symmetric matrix of Levenshtein distances between the phoneme lists of different dialects.

    Parameters:
        all_list (list of list of str): List of dialects, each containing a list of phoneme strings.

    Returns:
        np.ndarray: A square matrix of Levenshtein distances.
    """
    num_dialects = len(all_list)
    all_feats = np.zeros((num_dialects, num_dialects))
    
    for dialect_1 in range(num_dialects):
        for dialect_2 in range(dialect_1 + 1, num_dialects):
            cost = 0
            for index in range(len(all_list[0])):
                str1, str2 = all_list[dialect_1][index], all_list[dialect_2][index]
                cost_, _ = Levenshtein_distance(str1, str2)
                cost += cost_
            all_feats[dialect_1][dialect_2] = all_feats[dialect_2][dialect_1] = cost
    
    return all_feats


def feats(all_list, tone_list):
    """
    Combines phoneme and tone features into a single feature set.

    Parameters:
        all_list (np.ndarray): Array containing phoneme features.
        tone_list (np.ndarray): Array containing tone features.

    Returns:
        np.ndarray: The concatenated features.
    """
    feats_1 = phoneme_feats(all_list)  # Assumes this function is defined elsewhere and returns an np.ndarray
    feats_2 = tone_feats(tone_list)  # Assumes this function is defined elsewhere and returns an np.ndarray
    feats = np.concatenate((feats_1, feats_2), axis=-1)
    return feats