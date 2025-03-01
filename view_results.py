import os

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datasets.get_test_image_list import get_test_list, get_pdefined_anchors, getImagePath
from py_utils import dir_utils, load_utils


def viewBBoxes(image_file, bboxes, titles, showImageName=True):

    n_items_per_row = 4
    image = Image.open(image_file)
    image = np.array(image, dtype=np.uint8)
    n_crops = len(bboxes)
    n_rows = n_crops // n_items_per_row + 1

    fig = plt.figure(figsize=[20, 20])
    if showImageName:
        fig.suptitle(os.path.basename(image_file))

    for idx, s_bbox in enumerate(bboxes):
        ax =fig.add_subplot(n_rows, n_items_per_row, idx+1)
        ax.imshow(image)
        ax.set_axis_off()

        ax.set_title(titles[idx])

        rect_i = patches.Rectangle((s_bbox[0], s_bbox[1]), s_bbox[2]-s_bbox[0], s_bbox[3]-s_bbox[1], linewidth=2, edgecolor='yellow', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_i)
    plt.show(block=False)
    eval(input("Press Enter to continue..."))
    plt.close()


annotation_path = '/home/zwei/Dev/adobe_pytorch_share0/snapshots/MTweak3-FullVGG-1024x512/MTweak3-FullVGG-1024x512.txt'
image_path_root = getImagePath()

image_data = load_utils.load_json(annotation_path)
for image_name in list(image_data.keys()):
    s_image_path = os.path.join(image_path_root, image_name)
    bboxes = image_data[image_name]['bboxes']
    scores = image_data[image_name]['scores']
    viewBBoxes(s_image_path, bboxes, scores)

print("DEBUG")

