import random
import shutil
import os
import cv2
import time
import yaml
import csv
import datetime
import torch
import argparse

from tqdm import tqdm

from my_utils import recreate_folder, get_all_files_in_folder, get_last_exp_number


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
                   nms: float,
                   test_split_part: float) -> int:
    #
    data_dir = os.path.join(source_folder_class, "data")
    txts = get_all_files_in_folder(data_dir, ["*.txt"])

    if len(txts) >= min_samples_count:
        prepare_for_training(data_dir, images_ext, split_part=test_split_part)

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


def pseudolabeling(data_dir: str,
                   weights: str,
                   threshold: float,
                   nms: float,
                   image_size: int,
                   images_ext: str) -> None:
    #
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


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('classes_file', type=str, default='obj.names', help='List of classes')
    parser.add_argument('images_ext', type=str, default='jpg', help='Exstension of images')

    # Training params
    parser.add_argument('class_for_training', type=str, help='Name of class for training')
    parser.add_argument('weights', type=str, help='Path to pretraining weights')
    parser.add_argument('--min_samples_count', type=int, default=200, help='Min count of samples to start training')
    parser.add_argument('--image_size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--test_split_part', type=float, default=0.2)
    parser.add_argument('--min_map', type=float, default=0.95, help='Min mAP@:.5:.95 for pseudolabeling')
    parser.add_argument('--sleep_training_sec', '--sleep', type=int, default=20,
                        help='Wait before next training attempt (min)')
    parser.add_argument('--max_attempts', type=int, default=10, help='Max count of attempts')

    # Inference params
    parser.add_argument('--threshold', '--th', type=float, default=0.8, help='model threshold')
    parser.add_argument('--nms', type=float, default=0.5, help='model nms')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    # project_name = "door_smoke"

    classes_file = opt.classes_file
    # classes_file = "obj.names"
    classes_file_path = f"data/{project_name}/{classes_file}"
    with open(classes_file_path) as file:
        classes = [line.rstrip() for line in file]

    dataset_path = f"data/{project_name}/dataset"
    source_folder = f"data/{project_name}/labeling"

    # training params
    class_for_training = opt.class_for_training
    # class_for_training = "door"
    images_ext = opt.images_ext
    # images_ext = "jpg"
    source_folder_class = os.path.join(source_folder,
                                       str(classes.index(class_for_training)) + "_" + class_for_training)

    min_samples_count = opt.min_samples_count
    # min_samples_count = 200

    image_size = opt.image_size
    # image_size = 640

    batch_size = opt.batch_size
    # batch_size = 16

    epochs = opt.epochs
    # epochs = 7

    weights = opt.weights
    # weights = "yolov5_weights/yolov5m.pt"

    min_mAP_095 = opt.min_map
    # min_mAP_095 = 0.5

    sleep_training_sec = opt.sleep_training_sec * 60
    # sleep_training_sec = 60 * 20

    test_split_part = opt.test_split_part
    # test_split_part = 0.2

    # Inference params
    threshold = opt.threshold
    # threshold = 0.5

    nms = opt.nms
    # nms = 0.3

    max_attempts = opt.max_attempts
    # max_attempts = 10

    attempt = 0
    while attempt < max_attempts:
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
                                           nms,
                                           test_split_part)
        attempt += attempt_increment
