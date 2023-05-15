import configparser

config = configparser.ConfigParser()
config.read("config.ini")
# Paths to datasets paths
MIDI_PATH = config["DATASET"]["midi"]
TRAIN_PATH = config["DATASET"]["train_path"]
TEST_PATH = config["DATASET"]["test_path"]


TRAIN_VECTOR_PATH = config["DATASET"]["vectorized_train_path"]
TSET_VECTOR_PATH = config["DATASET"]["vectorized_test_path"]

LMODEL_PATH = config["MODEL"]["language_model_path"] # Can be downloaded from https://www.kaggle.com/datasets/sandreds/googlenewsvectorsnegative300

EMBEDDING_SIZE = 300
MAX_SEQ_LENGTH = 7