import os
import shutil
import argparse
import yaml

from typing import List
from tqdm import tqdm


def replicate_dataset(classes: List,
                      dataset_path: str,
                      source_folder: str,
                      classes_file: str) -> bool:
    #
    if os.path.isdir(source_folder):
        print(
            f"Folder \"{source_folder}\" exist!\nIf you want recreate this folder, "
            f"please, remove folder \"{source_folder}\" manually.")
        return False

    for cl in tqdm(classes, desc="Copying dataset ..."):
        shutil.copytree(dataset_path, os.path.join(os.sep.join(dataset_path.split(os.sep)[:-1]),
                                                   source_folder.split(os.sep)[-1],
                                                   str(classes.index(cl)) + "_" + cl, "data"))

        with open(os.path.join(os.sep.join(dataset_path.split(os.sep)[:-1]),
                               source_folder.split(os.sep)[-1],
                               str(classes.index(cl)) + "_" + cl, classes_file), 'w') as f:
            f.write("%s\n" % (cl))

    return True


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name

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

    replicate_dataset(classes, dataset_path, source_folder, classes_file)
