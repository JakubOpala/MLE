import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import mlflow
mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['test']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".pth"):
                if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pth') < \
                        datetime.strptime(filename, conf['general']['datetime_format'] + '.pth'):
                    latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_model_by_path(path: str) -> nn.Module:
    """Loads and returns the specified PyTorch model"""
    try:        
        # Load the entire model
        model = torch.load(path)
        #nn.Sequential.load_state_dict(torch.load(path))
        
        logging.info(f'Path of the model: {path}')
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)

def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)

def get_latest_scaler_path() -> str:
    """Gets the path of the latest saved StandardScaler"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".joblib"):
                if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.joblib') < \
                        datetime.strptime(filename, conf['general']['datetime_format'] + '.joblib'):
                    latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_scaler_by_path(path: str):
    """Loads and returns the specified StandardScaler"""
    try:
        scaler = joblib.load(path)
        logging.info(f'Path of the scaler: {path}')
        return scaler
    except Exception as e:
        logging.error(f'An error occurred while loading the scaler: {e}')
        sys.exit(1)

def predict_results(model, scaler, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join it with the infer_data"""

    infer_data_scaled = pd.DataFrame(scaler.transform(infer_data), columns=infer_data.columns)

    # Convert DataFrame to PyTorch tensor
    infer_data_tensor = torch.tensor(infer_data_scaled.values, dtype=torch.float32)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(infer_data_tensor)

    # Get the class indices with the highest probability for each prediction
    predicted_indices = torch.argmax(predictions, dim=1)

    # Assuming you have a list of class labels like ['class1', 'class2', 'class3']
    class_labels = ['setosa','versicolor','virginica']

    # Map the predicted indices to class labels
    predicted_classes = [class_labels[idx] for idx in predicted_indices]

    # Add the predicted classes to the DataFrame
    infer_data['Predicted_species'] = predicted_classes

    return infer_data

def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    scaler = get_scaler_by_path(get_latest_scaler_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, scaler, infer_data.drop(columns=['Species']))
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()