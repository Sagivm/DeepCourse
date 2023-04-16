
from Ex2.util.format_dataset import *
from Ex2 import *
from Ex2.siamese_model import build_base_model
# def get_images(path):
#     names = os.listdir(path)
#     images = []
#     for name in names:
#         name_path = os.path.join(path,name)
#         images += [f"{name}/{name_image}" for name_image in os.listdir(name_path)]
#
#     df = pd.DataFrame(images,columns=None)
#     df.to_csv('list.csv', index=False)
#
#
# def count_items_in_subdir(root_dir):
#     t = list()
#     for subdir, dirs, files in os.walk(root_dir):
#         #print(f"{subdir}: {len(files)} items")
#         t.append([subdir.split("/")[-1],len(files)])
#     df = pd.DataFrame(t, columns=None)
#     df.to_csv('list.csv', index=False)
# path = "../data/lfw2/lfw2/"
# get_images(path)
# count_items_in_subdir(path)

def get_dataset():
    format_dataset(TRAIN_PATH, DATASET_PATH)
    format_dataset(TEST_PATH, DATASET_PATH)
    model = build_base_model((105,105,1))
    print(model.summary())

get_dataset()