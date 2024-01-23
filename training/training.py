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
import itertools
import copy

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
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])    

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df
    
    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, 
                           random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df
    

class Training():
    def __init__(self, df , test_size=0.2) -> None:
        
        self.model = None
        data = self.data_split(df, test_size)
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]
        #self.build_model(layer_sizes, regularization,activation, dropout_rate)

    def build_model(self, layer_sizes, regularization, 
                    activation, dropout_rate=0.2):
        
        layers = []
        for i in range(len(layer_sizes) - 2):
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            
            # assign most suitable initialization for activation functions
            if activation == 'relu':
                nn.init.kaiming_normal_(linear_layer.weight, 
                                        mode='fan_in', nonlinearity='relu')
            elif activation == 'tanh':
                nn.init.xavier_uniform_(linear_layer.weight)

            # Add the initialized linear layer to the model
            layers.append(linear_layer)
            
            if regularization == 'batch_normalization':
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            elif regularization == 'dropout':
                layers.append(nn.Dropout(p=dropout_rate))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
        linear_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_uniform_(linear_layer.weight)
        layers.append(linear_layer)
        layers.append(nn.Softmax())
        model = nn.Sequential(*layers)
        return model

    def train_model(self, model, n_epochs, batch_size,
                    loss_fn, optimizer, patience=5):

        best_loss = np.inf   
        best_weights = None
        history_val = []
        history_train = []
        early_stopping_counter = 0

        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0.0
            
            for start in range(0, len(self.X_train), batch_size):
                end = min(start + batch_size, len(self.X_train))
                X_batch = self.X_train[start:end]
                y_batch = self.y_train[start:end]

                optimizer.zero_grad()

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
        
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * (end - start + 1)

            average_train_loss = total_train_loss / len(self.X_train)
            model.eval()
            history_train.append(average_train_loss)

            y_pred = model(self.X_test)
            val_loss = loss_fn(y_pred, self.y_test)
            val_loss = float(val_loss)
            history_val.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                break

        # Restore the best model
        model.load_state_dict(best_weights)

        return model, best_loss

    def test_models(self, device, architectures, regularizations, activations,
                    dropout_rates, learning_rates, batch_sizes, n_epochs=30):

        logging.info("Training models...")

        min_loss = float('inf')
        
        all_parameters = list(itertools.product(architectures, regularizations,
                                                activations, learning_rates,
                                                batch_sizes))

        for parameters in all_parameters:
            layers, regularization, activation, lr, batch_size = parameters
            if regularization == 'dropout':
                for dropout_rate in dropout_rates:
                    model = self.build_model(layers, regularization, 
                                            activation, dropout_rate)
                    model = model.to(device)
                    batch_start = torch.arange(0, len(self.X_train), batch_size)
                    loss_fn = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    model,  loss = self.train_model(model, n_epochs, batch_size,
                                                    loss_fn, optimizer)
                    '''
                    logging.info(f"Architecture: {architecture}, "
                                 f"regularization: {regularization}, "
                                 f"activation: {activation}, "
                                 f"dropout_rate: {dropout_rate}")
                    '''
                    if loss < min_loss:
                        min_loss = loss
                        best_model = model
            else:
                model = self.build_model(layers, regularization, activation)
                model = model.to(device)
                batch_start = torch.arange(0, len(self.X_train), batch_size)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                model,  loss = self.train_model(model, n_epochs, batch_size,
                                                loss_fn, optimizer)
                '''
                logging.info(f"Architecture: {architecture}, "
                                 f"regularization: {regularization}, "
                                 f"activation: {activation}, ")
                '''
                if loss < min_loss:
                    min_loss = loss
                    best_model = model
            
        logging.info(f"Best model loss: {min_loss}")
        self.save_model(best_model)
        return best_model

    def save_model(self, model, path=None):

        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            add_path = datetime.now().strftime(conf['general']['datetime_format'])+'.pth'
            path = os.path.join(MODEL_DIR, add_path)
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(model.state_dict(), path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple:
        logging.info("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(df[['x1','x2']], df['y'], 
                                                            test_size=test_size, 
                                random_state=conf['general']['random_state'])
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1,1)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1,1)

        return X_train, X_test, y_train, y_test
    





if __name__ == "__main__":
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data()
    tr.run_training(df, test_size=conf['train']['test_size'])

    architectures    = conf['train']['architectures']
    regularizations  = conf['train']['regularizations']
    activations      = conf['train']['activations']
    dropout_rates    = conf['train']['dropout_rates']
    learning_rates   = conf['train']['learning_rates']
    batch_sizes      = conf['train']['batch_sizes']
    
    tr.test_models(device, architectures, regularizations, activations,
                    dropout_rates, learning_rates, batch_sizes, n_epochs=30)

    
    
