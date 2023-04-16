import configparser

config = configparser.ConfigParser()
config.read("config.ini")
DATASET_PATH = config["DATASET"]["dataset_path"]
TRAIN_PATH = config["DATASET"]["train_pairs_path"]
TEST_PATH = config["DATASET"]["test_pairs_path"]