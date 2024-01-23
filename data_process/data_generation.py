import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys
import os
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging, singleton

url = "https://en.wikipedia.org/wiki/Iris_flower_data_set"

CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

@singleton
class Iris_generator():

    def __init__(self) -> None:
        self.df = None

    def load(self, url):
        logger.info("Downloading iris dataset...")
        dfs = pd.read_html(url, header=0)
        self.df = dfs[0]
    
    def train_test_split(self, save_train_path, save_inf_path, test_size=0.2):
        X = self.df.iloc[:, :-1]  # Features
        y = self.df.iloc[:, -1] 
        X_train, y_train, X_inference, y_inference = train_test_split(X, y, test_size)
        train_df = pd.concat([X_train, y_train], axis=1)
        inference_df = pd.concat([X_inference, y_inference], axis=1)
        self.save(train_df, save_train_path)
        self.save(inference_df, save_inf_path)

    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)
    


# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = Iris_generator()
    gen.load(url=url)
    gen.train_test_split(save_train_path=TRAIN_PATH, save_inf_path=INFERENCE_PATH, test_size=conf['train']['test_size'])
    logger.info("Script completed successfully.")