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

from labeling import pseudolabeling
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


def train(source_folder_class: str,
          images_ext: str,
          min_samples_count: int,
          model_input_image_size: int,
          batch_size: int,
          max_epochs: int,
          min_epochs: int,
          weights: str,
          test_split_part: float,
          min_mAP_095: float,
          count_of_epochs_min_map: int,
          resume_weights: str,
          resume_epochs: int) -> float:
    #
    mAP_095 = 0.0
    data_dir = os.path.join(source_folder_class, "data")
    txts = get_all_files_in_folder(data_dir, ["*.txt"])

    if len(txts) >= min_samples_count:
        prepare_for_training(data_dir, images_ext, split_part=test_split_part)

        # train
        yaml_path = os.path.join(source_folder_class, "training", "train.yml")
        project_path = os.path.join(source_folder_class, "training", "runs")
        logs_path = os.path.join(source_folder_class, "training", "logs")

        if resume_weights:
            os.system(
                f"python yolov5/train.py "
                f"--img {model_input_image_size} "
                f"--batch {batch_size} "
                f"--epochs {max_epochs + resume_epochs} "
                f"--data {yaml_path} "
                # f"--weights {weights} "
                f"--project {project_path} "
                f"--min_map {min_mAP_095} "
                f"--count_of_epochs_min_map {count_of_epochs_min_map} "
                f"--min_epochs {min_epochs} "
                f"--resume {resume_weights}")
        else:
            os.system(
                f"python yolov5/train.py "
                f"--img {model_input_image_size} "
                f"--batch {batch_size} "
                f"--epochs {max_epochs} "
                f"--data {yaml_path} "
                f"--weights {weights} "
                f"--project {project_path} "
                f"--min_map {min_mAP_095} "
                f"--count_of_epochs_min_map {count_of_epochs_min_map} "
                f"--min_epochs {min_epochs}")

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

    return mAP_095


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('class_for_training', type=str, help='Name of class for training')
    parser.add_argument('--count_of_images_to_markup', '--img_count', type=str,
                        help='Count of images for labeling')
    parser.add_argument('--min_map', type=float,
                        help='Min mAP for pseudolabeling')
    parser.add_argument('--resume_weights', type=str)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    class_for_training = opt.class_for_training
    count_of_images_to_markup = opt.count_of_images_to_markup
    min_mAP_095 = opt.min_map
    resume_weights = opt.resume_weights

    config_file = f"data/{project_name}/config.yaml"

    with open(config_file, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            print(config_dict)
        except yaml.YAMLError as exc:
            print(exc)

    classes_file = config_dict['classes_file']
    classes_file_path = f"data/{project_name}/{classes_file}"
    with open(classes_file_path) as file:
        classes = [line.rstrip() for line in file]

    dataset_path = f"data/{project_name}/dataset"
    source_folder = f"data/{project_name}/labeling"

    # training params
    images_ext = config_dict['image_exstension']
    source_folder_class = os.path.join(source_folder,
                                       str(classes.index(class_for_training)) + "_" + class_for_training)

    min_samples_count = config_dict['min_samples_count']
    model_input_image_size = config_dict['model_input_image_size']

    batch_size = config_dict['batch_size']
    max_epochs = config_dict['max_epochs']
    min_epochs = config_dict['min_epochs']
    init_weights = config_dict['init_weights_path']
    if not min_mAP_095:
        min_mAP_095 = config_dict['min_map']
    sleep_training_sec = config_dict['sleep_training_min'] * 60
    test_split_part = config_dict['test_split_part']
    max_training_attempts = config_dict['max_training_attempts']
    count_of_epochs_min_map = config_dict['count_of_epochs_min_map']
    resume_epochs = config_dict['resume_epochs']

    # Inference params
    threshold = config_dict['threshold']
    nms = config_dict['nms']
    if not count_of_images_to_markup:
        count_of_images_to_markup = config_dict['number_of_images_to_markup']

    attempt = 0
    while attempt < max_training_attempts:
        mAP_095 = train(source_folder_class,
                        images_ext,
                        min_samples_count,
                        model_input_image_size,
                        batch_size,
                        max_epochs,
                        min_epochs,
                        init_weights,
                        test_split_part,
                        min_mAP_095,
                        count_of_epochs_min_map,
                        resume_weights,
                        resume_epochs)

        if mAP_095 >= min_mAP_095:
            project_path = os.path.join(source_folder_class, "training", "runs")
            last_exp_number = get_last_exp_number(project_path)
            pseudolabeling(data_dir=os.path.join(source_folder_class, "data"),
                           weights=os.path.join(source_folder_class, "training", "runs", "exp" + str(last_exp_number),
                                                "weights", "best.pt"),
                           threshold=threshold,
                           nms=nms,
                           model_input_image_size=model_input_image_size,
                           images_ext=images_ext,
                           count_of_images_to_markup=count_of_images_to_markup)
            attempt = max_training_attempts
        else:
            print(f"Current time: {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')}")
            time.sleep(sleep_training_sec)
            attempt += 1
