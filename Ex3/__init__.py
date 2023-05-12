import configparser

config = configparser.ConfigParser()
config.read("config.ini")
# Paths to datasets paths
MIDI_PATH = config["DATASET"]["midi"]
TRAIN_PATH = config["DATASET"]["train_path"]
TEST_PATH = config["DATASET"]["test_path"]


TRAIN_VECTOR_PATH = config["DATASET"]["vectorized_train_path"]
TSET_VECTOR_PATH = config["DATASET"]["vectorized_test_path"]

LMODEL_PATH = config["MODEL"]["language_model_path"]
# Hyper parameters for the model
# BATCH_SIZE = int(config["MODEL"]["batch_size"])
# EPOCHS = int(config["MODEL"]["epochs"])

MAX_SEQ_LENGTH = 5