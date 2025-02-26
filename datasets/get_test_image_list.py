import glob
import os
from py_utils import load_utils
import pickle


def getImagePath():
    user_root = os.path.expanduser('~')

    image_path = 'iaa/ViewEvaluationNet/datasets/created_dataset'
    return os.path.join(user_root, image_path)

# updated function iterates a directory instead of reading a text file
# and returns a list of paths
def get_test_list(directory) -> list[str]:
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

    return sorted(image_list)


def get_pdefined_anchors():
    user_root = os.path.expanduser('~')
    pdefined_anchor_file = 'Dev/adobe_pytorch/datasets/pdefined_anchor.pkl'
    pdefined_anchors = pickle.load(open(os.path.join(user_root, pdefined_anchor_file), 'r'))
    return pdefined_anchors
