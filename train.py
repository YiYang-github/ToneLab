
"""
train.py
train the models
"""

import os
import umap
import math
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import List
from torch import optim
from random import sample
from google.colab import drive
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, fcluster
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def select_region(region_list, num=7):
    """
    Randomly selects regions for training, validation, and testing.

    Args:
    - region_list (list): List of all available regions.
    - num (int): Number of regions to select for training.

    Returns:
    - tuple: Lists of training, validation, and testing regions.
    """
    train_regions = sample(region_list, num)
    remaining_regions = [region for region in region_list if region not in train_regions]
    valid_regions = sample(remaining_regions, 2)
    test_regions = [region for region in remaining_regions if region not in valid_regions]

    print("Training Regions:")
    print(", ".join(train_regions))
    print("\nValid Regions:")
    print(", ".join(valid_regions))
    print("\nTest Regions:")
    print(", ".join(test_regions))

    return train_regions, valid_regions, test_regions

def select_region_list(dict_dialect, region_list, train_region=None, valid_region=None, test_region=None, type="Chosen", num=7):
    """
    Selects and prints out the dialects from the specified regions based on the chosen mode.

    Args:
    - dict_dialect (dict): Dictionary mapping regions to dialects.
    - region_list (list): List of all regions.
    - train_region (list): Preselected training regions.
    - valid_region (list): Preselected validation regions.
    - test_region (list): Preselected testing regions.
    - type (str): Selection type ("Chosen" or other).
    - num (int): Number of regions to select for training if not preselected.

    Returns:
    - tuple: Lists of selected dialects for training, validation, and testing.
    """
    if type == "Chosen" and train_region and valid_region and test_region:
        train_regions = list(train_region)
        valid_regions = list(valid_region)
        test_regions = list(test_region)
    else:
        train_regions, valid_regions, test_regions = select_region(region_list, num=num)

    # Display regions
    print("Training Regions:")
    print(", ".join(train_regions))
    print("\nValid Regions:")
    print(", ".join(valid_regions))
    print("\nTest Regions:")
    print(", ".join(test_regions))

    # Compile selected dialects from the regions
    train_selected = [dict_dialect[area] for area in train_regions]
    valid_selected = [dict_dialect[area] for area in valid_regions]
    test_selected = [dict_dialect[area] for area in test_regions]

    # Output the count of selected dialects
    print(f"\nTrain Selected {len(train_regions)} regions with: {len(train_selected)} elements")
    print(f"Valid Selected {len(valid_regions)} regions with: {len(valid_selected)} elements")
    print(f"Test Selected {len(test_regions)} regions with: {len(test_selected)} elements\n")

    return train_selected, valid_selected, test_selected


def train_and_evaluate(model, train_loader, valid_loader, optimizer, device, num_epochs, save_dir, unique_transcription, print_interval):
    """
    Train the model on the training set and select the best performing model on the validation set.

    Args:
    - model: The neural network model to train.
    - train_loader: DataLoader for the training dataset.
    - valid_loader: DataLoader for the validation dataset.
    - optimizer: Optimizer for updating the model parameters.
    - device: Device to run the training on (e.g., 'cpu' or 'cuda').
    - num_epochs: Number of training epochs.
    - save_dir: Directory to save the best model and training logs.
    - unique_transcription: Tensor containing unique transcriptions for evaluation.
    - print_interval: int, interval for printing the logs.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_path = os.path.join(save_dir, 'training_log.txt')
    unique_transcription = unique_transcription.to(device)
    criterion = nn.L1Loss()

    with open(log_file_path, 'w') as log_file:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        best_valid_accuracy = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_accum = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                values = model(batch_X)
                loss = criterion(values, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()

            scheduler.step()
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_accum / len(train_loader):.4f}\n')

            model.eval()
            train_correct, valid_correct = 0, 0

            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    values = model(batch_X)
                    train_correct += eval_metric(values, batch_y, unique_transcription, metric_type='acc')

                for batch_X, batch_y in valid_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    values = model(batch_X)
                    valid_correct += eval_metric(values, batch_y, unique_transcription, metric_type='acc')

            valid_accuracy = valid_correct / len(valid_loader.dataset)
            train_accuracy = train_correct / len(train_loader.dataset)

            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n')

            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n')

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                log_file.write(f"Saved Best Model at Epoch {epoch+1} with Valid Accuracy: {best_valid_accuracy:.4f}\n")

            log_file.flush()

def testing(model, test_feat, test_label, unique_transcription, checkpoint_dir, device):
    """
    Evaluate the model on the test set.

    Args:
    - model: The neural network model to train.
    """

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model_state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(model_state_dict)

    unique_transcription = unique_transcription.to(device)

    model.eval()
    with torch.no_grad():
        feats, labels = test_feat.to(device), test_label.to(device)
        values = model(feats)

        accuracy = eval_metric(values, labels, unique_transcription, metric_type='acc')
        loss   = eval_metric(values, labels, unique_transcription, metric_type='mae')

        return accuracy/test_feat.shape[0], loss/test_feat.shape[0]


def eval_metric(values, batch_y, unique_transcription, metric_type='acc'):
    """
    Calculate the accuracy or MAE of the predictions.

    Args:
    - values: torch.tensor of shape (batch_size, 3), the predicted values.
    - batch_y: torch.tensor of shape (batch_size, 3), the true labels.
    - unique_transcription: torch.tensor of shape (n, 3), the tensor containing unique transcriptions.
    - metric_type: str, the type of metric to calculate ('acc' for accuracy, 'mae' for mean absolute error).

    Returns:
    - int or float, the calculated metric (number of correct predictions for accuracy, total MAE for mean absolute error).
    """
    correct = 0
    total_loss = 0.0

    for index in range(batch_y.size(0)):
        signal = batch_y[index]
        value = values[index]
        pred_seq = match_transcription(value, unique_transcription)

        with torch.no_grad():
            if torch.sum(torch.abs(pred_seq - signal)) < 1e-2:
                correct += 1
            total_loss += torch.sum(torch.abs(pred_seq - signal))

    if metric_type == 'acc':
        return correct
    elif metric_type == 'mae':
        return total_loss

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






