import configparser

config = configparser.ConfigParser()
config.read("config.ini")
# Paths to datasets paths
DATASET_PATH = config["DATASET"]["dataset_path"]
TRAIN_PATH = config["DATASET"]["train_pairs_path"]
TEST_PATH = config["DATASET"]["test_pairs_path"]

# Hyper parameters for the model
BATCH_SIZE = int(config["MODEL"]["batch_size"])
EPOCHS = int(config["MODEL"]["epochs"])