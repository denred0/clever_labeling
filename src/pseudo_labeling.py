import random
import shutil
import os
import cv2
import time
import yaml
import csv
import datetime
import torch

from typing import List
from tqdm import tqdm

from my_utils import recreate_folder, get_all_files_in_folder, get_last_exp_number


def replicate_dataset(classes: List,
                      dataset_path: str,
                      source_folder: str) -> bool:
    #
    if os.path.isdir(source_folder):
        print(
            f"Folder \"{source_folder}\" exist!\nIf you want recreate this folder, please, remove folder \"{source_folder}\" manually.")
        return False

    for cl in tqdm(classes, desc="Copying dataset ..."):
        shutil.copytree(dataset_path,
                        os.path.join(os.sep.join(dataset_path.split(os.sep)[:-1]),
                                     source_folder.split(os.sep)[-1],
                                     str(classes.index(cl)) + "_" + cl, "data"))

        with open(os.path.join(os.sep.join(dataset_path.split(os.sep)[:-1]),
                               source_folder.split(os.sep)[-1],
                               str(classes.index(cl)) + "_" + cl, "obj.names"), 'w') as f:
            for item in classes:
                f.write("%s\n" % (item))

    return True


def prepare_for_training(data_dir: str,
                         images_ext: str,
                         split_part=0.2) -> None:
    # create folders
    training_dir = os.path.join(os.sep.join(data_dir.split(os.sep)[:-1]), "training")
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)
    recreate_folder(os.path.join(training_dir, "train"))
    recreate_folder(os.path.join(training_dir, "val"))
    if not os.path.isdir(os.path.join(training_dir, "runs")):
        os.mkdir(os.path.join(training_dir, "runs"))
    if not os.path.isdir(os.path.join(training_dir, "logs")):
        os.mkdir(os.path.join(training_dir, "logs"))

    # copy to train/val
    txts = get_all_files_in_folder(data_dir, ["*.txt"])
    random.shuffle(txts)
    train_count = int(len(txts) * (1 - split_part))

    for i, txt in enumerate(txts):
        if i < train_count:
            shutil.copy(txt, os.path.join(training_dir, "train"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "train"))
        else:
            shutil.copy(txt, os.path.join(training_dir, "val"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "val"))

    # create training yaml
    train_yaml = dict(
        path=os.path.abspath(training_dir),
        train="train",
        val="val",
        nc=1,
        names=[f'{data_dir.split(os.sep)[-2].split("_")[1]}']
    )

    with open(os.path.join(training_dir, 'train.yml'), 'w') as outfile:
        yaml.dump(train_yaml, outfile, default_flow_style=None)


def start_training(source_folder_class: str,
                   images_ext: str,
                   min_samples_count: int,
                   image_size: int,
                   batch_size: int,
                   epochs: int,
                   weights: str,
                   min_mAP_095: float,
                   sleep_training_sec: int,
                   threshold: float,
                   nms: float) -> int:
    #
    data_dir = os.path.join(source_folder_class, "data")
    txts = get_all_files_in_folder(data_dir, ["*.txt"])

    if len(txts) >= min_samples_count:
        prepare_for_training(data_dir, images_ext, split_part=0.2)

        # train
        yaml_path = os.path.join(source_folder_class, "training", "train.yml")
        project_path = os.path.join(source_folder_class, "training", "runs")
        logs_path = os.path.join(source_folder_class, "training", "logs")

        os.system(
            f"python yolov5/train.py "
            f"--img {image_size} "
            f"--batch {batch_size} "
            f"--epochs {epochs} "
            f"--data {yaml_path} "
            f"--weights {weights} "
            f"--project {project_path}")

        # check results
        results = []
        last_exp_number = get_last_exp_number(project_path)
        with open(os.path.join(project_path, "exp" + str(last_exp_number), "results.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                results.append(row)

        mAP_095 = float(results[-1][7].strip())
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        open(os.path.join(logs_path,
                          timestamp + "_mAP_" + str(round(mAP_095, 4))
                          + "_img_count_" + str(len(txts)) + ".txt"), 'a').close()

        if mAP_095 >= min_mAP_095:
            pseudolabeling(data_dir=data_dir,
                           weights=os.path.join(source_folder_class, "training", "runs", "exp" + str(last_exp_number),
                                                "weights", "best.pt"),
                           threshold=threshold,
                           nms=nms,
                           image_size=image_size,
                           images_ext=images_ext)
            return 1000
        else:
            time.sleep(sleep_training_sec)
            return 1
    else:
        time.sleep(sleep_training_sec)
        return 1


def pseudolabeling(data_dir, weights, threshold, nms, image_size, images_ext):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
    model.conf = threshold
    model.iou = nms
    model.classes = [int(data_dir.split(os.sep)[-2].split("_")[0])]

    images = get_all_files_in_folder(data_dir, [f"*.{images_ext}"])
    images = [im.stem for im in images]
    txts = get_all_files_in_folder(data_dir, [f"*.txt"])
    txts = [txt.stem for txt in txts]

    images_for_labeling = list(set(images) - set(txts))
    images_for_labeling = [im + "." + images_ext for im in images_for_labeling]

    for im in tqdm(images_for_labeling):
        img = cv2.imread(os.path.join(data_dir, im), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        detections = model(img, size=image_size)
        results_list = detections.pandas().xyxy[0].values.tolist()

        detections_valid = [d for d in results_list if float(d[4]) > threshold]
        detections_result = []
        for res in detections_valid:
            (xmin, ymin) = (res[0], res[1])
            (xmax, ymax) = (res[2], res[3])
            width = xmax - xmin
            height = ymax - ymin

            x_center_norm = float((xmax - xmin) / 2 + xmin) / w
            y_center_norm = float((ymax - ymin) / 2 + ymin) / h
            w_norm = float(width) / w
            h_norm = float(height) / h

            if w_norm > 1: w_norm = 1.0
            if h_norm > 1: h_norm = 1.0

            detections_result.append(
                [
                    res[5],
                    x_center_norm,
                    y_center_norm,
                    w_norm,
                    h_norm
                ])

        if detections_valid:
            with open(os.path.join(data_dir, im.split(".")[0] + ".txt"), 'w') as f:
                for item in detections_result:
                    f.write("%s\n" % (str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(
                        item[3]) + ' ' + str(item[4])))


if __name__ == "__main__":
    project_name = "door_smoke"
    classes_file = f"data/{project_name}/obj.names"
    with open(classes_file) as file:
        classes = [line.rstrip() for line in file]

    dataset_path = f"data/{project_name}/dataset"
    images_ext = "jpg"
    source_folder = f"data/{project_name}/labeling"

    need_prepare_dataset = False
    if need_prepare_dataset:
        if not replicate_dataset(classes, dataset_path, source_folder):
            exit()

    # training params
    class_for_training = "door"
    source_folder_class = os.path.join(source_folder,
                                       str(classes.index(class_for_training)) + "_" + class_for_training)
    min_samples_count = 200
    image_size = 640
    batch_size = 16
    epochs = 7
    weights = "yolov5_weights/yolov5m.pt"
    min_mAP_095 = 0.5
    sleep_training_sec = 60 * 20

    # model params
    threshold = 0.5
    nms = 0.3

    attempt = 0
    max_attempts = 10
    while attempt < 10:
        attempt_increment = start_training(source_folder_class,
                                           images_ext,
                                           min_samples_count,
                                           image_size,
                                           batch_size,
                                           epochs,
                                           weights,
                                           min_mAP_095,
                                           sleep_training_sec,
                                           threshold,
                                           nms)
        min_samples_count += 100
        attempt += attempt_increment
