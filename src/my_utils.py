import random
import os
import shutil
import numpy as np
import torch
import cv2
import yaml
from tqdm import tqdm

from typing import List
from pathlib import Path


def get_all_files_in_folder(folder: str, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def recreate_folders(root_dir: Path, folders_list: List) -> None:
    for directory in folders_list:
        output_dir = root_dir.joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def recreate_folder(root_dir: str) -> None:
    output_dir = Path(root_dir)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_last_exp_number(root_folder):
    folders = [x[0] for x in os.walk(root_folder)][1:]
    folders = [x.split("/")[-1] for x in folders]
    folders_exp = []
    for f in folders:
        if "exp" in f:
            f = f.replace("exp", "")
            if f == "": f = "0"
            folders_exp.append(f)

    if len(folders_exp) == 1 or len(folders_exp) == 0:
        return ""
    else:
        return max([int(x) for x in folders_exp])


def plot_one_box(im, box, label=None, color=(255, 255, 0), line_thickness=1, write_label=True):
    c1 = (box[0], box[1])
    c2 = (box[2], box[3])

    tl = line_thickness or round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    im = cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label and write_label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        im = cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        im = cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im


def read_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            # print(config_dict)
        except yaml.YAMLError as exc:
            print(exc)

    return config_dict


def intersection_over_union_box(box_pred, box_label, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = box_pred[0] - box_pred[2] / 2
        box1_y1 = box_pred[1] - box_pred[3] / 2
        box1_x2 = box_pred[0] + box_pred[2] / 2
        box1_y2 = box_pred[1] + box_pred[3] / 2
        box2_x1 = box_label[0] - box_label[2] / 2
        box2_y1 = box_label[1] - box_label[3] / 2
        box2_x2 = box_label[0] + box_label[2] / 2
        box2_y2 = box_label[1] + box_label[3] / 2

    elif box_format == "corners":
        box1_x1 = box_pred[0]
        box1_y1 = box_pred[1]
        box1_x2 = box_pred[2]
        box1_y2 = box_pred[3]
        box2_x1 = box_label[0]
        box2_y1 = box_label[1]
        box2_x2 = box_label[2]
        box2_y2 = box_label[3]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    if x2 - x1 < 0 or y2 - y1 < 0:
        return 0
    else:
        intersection = np.clip(x2 - x1, 0, x2 - x1) * np.clip(y2 - y1, 0, y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return round(intersection / (box1_area + box2_area - intersection + 1e-6), 2)


def clean_txts():
    path = "data/evraz_attr/labeling/2_orange_helmet/data"
    imgs = get_all_files_in_folder(path, ["*.jpg"])
    imgs = [img.stem for img in imgs]
    txts = get_all_files_in_folder(path, ["*.txt"])
    txts = [txt.stem for txt in txts]

    deleted = 0
    for txt in tqdm(txts):
        if txt not in imgs:
            try:
                os.remove(os.path.join(path, txt + ".txt"))
                deleted += 1
            except OSError:
                pass

    print(f"Deleted: {deleted}")


if __name__ == "__main__":
    clean_txts()
