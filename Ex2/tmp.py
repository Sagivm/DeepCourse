from Ex2.util.format_dataset import format_dataset
from Ex2 import DATASET_PATH,TRAIN_PATH, TEST_PATH , BATCH_SIZE, EPOCHS
from Ex2.siamese_model import build_base_model, build_siamese_model, batch_generator
from keras.optimizers import SGD
import numpy as np
import imageio
from sklearn.metrics.pairwise import cosine_similarity

from numpy.linalg import norm


x=np.array(np.asarray(imageio.imread(f"{DATASET_PATH}/George_W_Bush/George_W_Bush_0001.jpg")).flat)
y=np.array(np.asarray(imageio.imread(f"{DATASET_PATH}/George_Ryan/George_Ryan_0002.jpg")).flat)
t=np.array([1,2,3])
print(cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)))

t=0