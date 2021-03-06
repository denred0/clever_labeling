import random
import shutil
import os
import time
from typing import List

import yaml
import csv
import numpy as np
import argparse
import pandas as pd

from tqdm import tqdm
from numpy import loadtxt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from labeling import pseudolabeling
from my_utils import recreate_folder, get_all_files_in_folder, get_last_exp_number, read_config

import warnings

warnings.filterwarnings("ignore")


def prepare_for_training(data_dir: str,
                         previous_exp: int,
                         full_train,
                         images_filter: List,
                         take_first_count: int,
                         take_last_count: int,
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

    # if images_filter:
    #     txts = []
    #     imgs = []
    #     for filter_value in images_filter:
    #         txts.extend(get_all_files_in_folder(data_dir, [f"*{filter_value}*.txt"]))
    #         imgs.extend(get_all_files_in_folder(data_dir, [f"*{filter_value}*.{images_ext}"]))
    # else:
    txts = get_all_files_in_folder(data_dir, ["*.txt"])
    imgs = get_all_files_in_folder(data_dir, [f"*.{images_ext}"])

    prev_txts = prev_imgs = []
    if previous_exp != -1:
        if not full_train:
            with open(os.path.join(training_dir, f"train_val{previous_exp}.txt")) as file:
                prev_txts = file.readlines()
                prev_txts = [line.rstrip() for line in prev_txts]
                prev_imgs = [txt.split('.')[0] + f".{images_ext}" for txt in prev_txts]

    txts = [txt for txt in txts if txt.name not in prev_txts]
    imgs = [img for img in imgs if img.name not in prev_imgs]

    print(f'Total images: {len(imgs)}')

    empty_txts = []
    non_empty_txts = []
    for txt in txts:
        lines = loadtxt(str(txt), delimiter=' ', unpack=False)
        if list(lines):
            non_empty_txts.append(txt)
        else:
            empty_txts.append(txt)

        # non empty txts
    # if take_first_count != -1 and len(non_empty_txts) > take_first_count:
    #     non_empty_txts = non_empty_txts[:take_first_count]
    #
    # if take_last_count != -1 and len(non_empty_txts) > take_last_count:
    #     non_empty_txts = non_empty_txts[-take_last_count:]

    random.shuffle(non_empty_txts)
    train_count = int(len(non_empty_txts) * (1 - split_part))

    for i, txt in tqdm(enumerate(non_empty_txts)):
        if i < train_count:
            shutil.copy(txt, os.path.join(training_dir, "train"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "train"))
        else:
            shutil.copy(txt, os.path.join(training_dir, "val"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "val"))

    # empty txts
    # if take_first_count != -1 and len(empty_txts) > take_first_count * 2:
    #     empty_txts = empty_txts[:(2 * take_first_count)]
    #
    # if take_last_count != -1 and len(empty_txts) > take_last_count * 2:
    #     empty_txts = empty_txts[-(2 * take_last_count):]

    random.shuffle(empty_txts)
    if len(empty_txts) > 2 * len(non_empty_txts):
        empty_txts = empty_txts[:2 * len(non_empty_txts)]
    train_count = int(len(empty_txts) * (1 - split_part))

    for i, txt in tqdm(enumerate(empty_txts)):
        if i < train_count:
            shutil.copy(txt, os.path.join(training_dir, "train"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "train"))
        else:
            shutil.copy(txt, os.path.join(training_dir, "val"))
            shutil.copy(str(txt.parent) + os.sep + txt.stem + "." + images_ext, os.path.join(training_dir, "val"))

    print(f"Labeled images: {len(non_empty_txts)}")
    print(f"Empty images: {len(empty_txts)}")
    print(f"Total training images: {len(empty_txts) + len(non_empty_txts)}")

    all_txts = [txt.name for txt in non_empty_txts] + [txt.name for txt in empty_txts] + prev_txts
    previous_exp = previous_exp + 1 if previous_exp != 0 else 2
    with open(os.path.join(training_dir, f'train_val{previous_exp}.txt'), 'w') as f:
        for item in all_txts:
            f.write("%s\n" % item)

    # create training yaml
    train_yaml = dict(
        path=os.path.abspath(training_dir),
        train="train",
        val="val",
        nc=1,
        names=[f'{data_dir.split(os.sep)[-2]}']
    )

    with open(os.path.join(training_dir, 'train.yml'), 'w') as outfile:
        yaml.dump(train_yaml, outfile, default_flow_style=None)


def train(source_folder_class: str,
          images_ext: str,
          min_samples_count: int,
          take_first_count: int,
          take_last_count: int,
          images_filter: List,
          model_input_image_size: int,
          batch_size: int,
          max_epochs: int,
          min_epochs: int,
          weights: str,
          test_split_part: float,
          min_mAP_095: float,
          count_of_epochs_min_map: int,
          resume_epochs: int,
          count_epochs_before_result: int,
          full_train) -> float:
    #
    mAP_095 = 0.0
    # train
    yaml_path = os.path.join(source_folder_class, "training", "train.yml")
    project_path = os.path.join(source_folder_class, "training", "runs")
    logs_path = os.path.join(source_folder_class, "training", "logs")
    last_exp_number = get_last_exp_number(project_path)

    data_dir = os.path.join(source_folder_class, "data")
    if images_filter:
        txts = []
        for filter_value in images_filter:
            txts.extend(get_all_files_in_folder(data_dir, [f"*{filter_value}*.txt"]))
    else:
        txts = get_all_files_in_folder(data_dir, ["*.txt"])

    prev_txts = []
    if last_exp_number != -1:
        if not full_train:
            with open(
                    os.path.join(os.path.join(source_folder_class, "training"),
                                 f"train_val{last_exp_number}.txt")) as file:
                prev_txts = file.readlines()
                prev_txts = [line.rstrip() for line in prev_txts]

    txts = [txt for txt in txts if txt.name not in prev_txts]

    non_empty_txts = []
    for txt in txts:
        lines = loadtxt(str(txt), delimiter=' ', unpack=False)
        if list(lines):
            non_empty_txts.append(txt)

    if len(non_empty_txts) >= min_samples_count or (len(prev_txts) and len(non_empty_txts)):
        prepare_for_training(data_dir,
                             last_exp_number,
                             full_train,
                             images_filter,
                             take_first_count,
                             take_last_count,
                             images_ext,
                             split_part=test_split_part)

        if last_exp_number == -1 or full_train:
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
                f"--min_epochs {min_epochs} "
                f"--count_epochs_before_result {count_epochs_before_result}")
        else:
            last_exp_number_str = str(last_exp_number) if last_exp_number != 0 else ''
            weights_path = os.path.join(source_folder_class, "training", "runs", f"exp{last_exp_number_str}", "weights",
                                        "last.pt")
            hyp = "yolov5/data/hyps/hyp.scratch-med.yaml"
            os.system(
                f"python yolov5/train.py "
                f"--img {model_input_image_size} "
                f"--batch {batch_size} "
                f"--epochs {5000} "
                f"--data {yaml_path} "
                f"--weights {weights_path} "
                f"--project {project_path} "
                f"--min_map {0} "
                f"--count_of_epochs_min_map {resume_epochs} "
                f"--min_epochs {resume_epochs} "
                f"--count_epochs_before_result {0} "
                f"--hyp {hyp}")

        # check results
        results = []
        last_exp_number = get_last_exp_number(project_path)
        with open(os.path.join(project_path, f"exp{last_exp_number}", "results.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                results.append(row)

        mAP_095 = float(results[-1][7].strip())
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        open(os.path.join(logs_path,
                          f"{timestamp}_exp{last_exp_number}_mAP_{round(mAP_095, 4)}_img_count_{len(txts)}.txt"),
             'a').close()

    else:
        print(f"Count of labeled images: {len(non_empty_txts)}/{min_samples_count}")

    return mAP_095


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('class_for_training', type=str, help='Name of class for training')
    parser.add_argument('--full_train', action="store_true", help='Training with all images')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    class_for_training = opt.class_for_training
    full_train = opt.full_train

    config_file = os.path.join("data", project_name, "labeling_config.yaml")
    labeling_config = read_config(config_file)
    max_training_attempts = labeling_config['max_training_attempts']

    attempt = 1
    while attempt < max_training_attempts:

        labeling_config = read_config(config_file)

        classes_file = labeling_config['classes_file']
        classes_file_path = os.path.join("data", project_name, classes_file)
        with open(classes_file_path) as file:
            classes = [line.rstrip() for line in file]

        dataset_path = f"data/{project_name}/dataset"
        source_folder = f"data/{project_name}/labeling"

        # training params
        images_ext = labeling_config['image_exstension']
        source_folder_class = os.path.join(source_folder,
                                           str(classes.index(class_for_training)) + "_" + class_for_training)

        min_samples_count = labeling_config['min_samples_count']
        model_input_image_size = labeling_config['model_input_image_size']

        batch_size = labeling_config['batch_size']
        max_epochs = labeling_config['max_epochs']
        min_epochs = labeling_config['min_epochs']
        init_weights = labeling_config['init_weights_path']
        min_mAP_095 = labeling_config['min_map']
        sleep_training_min = labeling_config['sleep_training_min']
        test_split_part = labeling_config['test_split_part']

        count_of_epochs_min_map = labeling_config['count_of_epochs_min_map']
        resume_epochs = labeling_config['resume_epochs']
        take_last_count = labeling_config['take_last_count']
        take_first_count = labeling_config['take_first_count']
        images_filter = labeling_config['images_filter']
        count_epochs_before_result = labeling_config['count_epochs_before_result']

        # Inference params
        threshold = labeling_config['threshold']
        nms = labeling_config['nms']
        count_of_images_to_markup = labeling_config['count_of_images_to_markup']

        mAP_095 = train(source_folder_class,
                        images_ext,
                        min_samples_count,
                        take_first_count,
                        take_last_count,
                        images_filter,
                        model_input_image_size,
                        batch_size,
                        max_epochs,
                        min_epochs,
                        init_weights,
                        test_split_part,
                        min_mAP_095,
                        count_of_epochs_min_map,
                        resume_epochs,
                        count_epochs_before_result,
                        full_train)

        if mAP_095 >= min_mAP_095:
            project_path = os.path.join(source_folder_class, "training", "runs")
            last_exp_number = get_last_exp_number(project_path)
            print("Training finished!")
            print(f"Current mAP {mAP_095} is greater than {min_mAP_095}")
            exp_path = os.path.join(source_folder_class, "training", "runs", "exp" + str(last_exp_number))
            print(f"Experiment path: {exp_path}")

            attempt = max_training_attempts
        else:
            now = datetime.now()
            next_training_time = now + timedelta(minutes=sleep_training_min)
            print(f"Attempt {attempt}/{max_training_attempts}")
            print(f"Current mAP {mAP_095} is below than {min_mAP_095}")
            print(f"Next train at: {next_training_time.strftime('%H:%M:%S')}\n")
            time.sleep(sleep_training_min * 60)
            attempt += 1
