"""
utils.py
"""

import math
import numpy as np

import torch

def print_model_details(model, model_name):
    """
    Print details about a PyTorch model including its name, total number of trainable parameters,
    and its structure.

    Parameters:
        model (torch.nn.Module): The PyTorch model to describe.
        model_name (str): A name or description for the model.

    Returns:
        None
    """
    print(f"Details for {model_name}:")
    print("-" * 40)
    # Calculate the total number of trainable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Print the structure of the model
    print("\nModel Structure:")
    print(model)
    print("-" * 100 + "\n")

# Example usage:
# Assuming `model` is a PyTorch model instance and `model_name` is a descriptive string for the model
# model = torch.nn.Linear(10, 5)  # Example model
# print_model_details(model, "Simple Linear Model")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth specified by latitude and longitude.

    Parameters:
    lat1 (float): Latitude of point 1 in degrees.
    lon1 (float): Longitude of point 1 in degrees.
    lat2 (float): Latitude of point 2 in degrees.
    lon2 (float): Longitude of point 2 in degrees.

    Returns:
    float: Distance between the two points in kilometers.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    # Haversine formula
    a = (math.sin(dLat / 2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate distance
    distance = R * c
    return distance


def check_null(_list):
    """
    Check if any sublist in a 2D list contains an empty list.

    Parameters:
        _list (list of list): The 2D list to check.

    Returns:
        bool: True if no empty lists are found, False otherwise.
    """
    for row in _list:
        if [] in row:
            return False
    return True

