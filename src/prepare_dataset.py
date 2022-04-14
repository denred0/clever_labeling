import os
import shutil
import argparse
import yaml
from collections import defaultdict

from typing import List
from tqdm import tqdm
from my_utils import read_config, get_all_files_in_folder


def replicate_dataset(classes: List,
                      dataset_path: str,
                      source_folder: str,
                      classes_file: str,
                      images_ext: str,
                      update_txts: bool) -> bool:
    #
    # if os.path.isdir(source_folder):
    #     print(
    #         f"Folder \"{source_folder}\" exist!\nIf you want recreate this folder, "
    #         f"please, remove folder \"{source_folder}\" manually.")
    #     return False

    images = get_all_files_in_folder(dataset_path, [f"*.{images_ext}"])

    for cl in tqdm(classes, desc="Copying dataset"):
        dest_folder = os.path.join(os.sep.join(dataset_path.split(os.sep)[:-1]), source_folder.split(os.sep)[-1],
                                   str(classes.index(cl)) + "_" + cl)

        # check if directory doesn't exist
        if not os.path.isdir(os.path.join(dest_folder, "data")):
            os.makedirs(os.path.join(dest_folder, "data"))

        for img in tqdm(images):
            shutil.copy(img, os.path.join(dest_folder, "data"))

        with open(os.path.join(dest_folder, classes_file), 'w') as f:
            f.write("%s\n" % (cl))

    if update_txts:
        txts = get_all_files_in_folder(dataset_path, ["*.txt"])

        for txt in tqdm(txts, desc="Recreating txts"):
            with open(txt) as txt_file:
                lines = [line.rstrip() for line in txt_file.readlines()]

            txt_dict = defaultdict(list)
            for line in lines:
                txt_dict[int(line.split()[0])].append(" ".join(line.split(" ")[1:]))

            for cl, val in txt_dict.items():
                new_txt_path = os.path.join("data", project_name, "labeling", str(cl) + "_" + classes[cl], "data",
                                            txt.name)

                if os.path.isfile(new_txt_path):
                    os.remove(new_txt_path)

                with open(new_txt_path, 'w') as f:
                    for line in val:
                        f.write("%s\n" % ("0 " + str(line)))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help="project's folder name")
    parser.add_argument('--upd_txts', action="store_true", help="update txts for classes")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    update_txts = opt.upd_txts

    config_file = os.path.join("data", project_name, "labeling_config.yaml")
    config_dict = read_config(config_file)

    classes_file = config_dict['classes_file']
    classes_file_path = os.path.join("data", project_name, classes_file)
    with open(classes_file_path) as file:
        classes = [line.rstrip() for line in file]

    dataset_path = os.path.join("data", project_name, "dataset")
    source_folder = os.path.join("data", project_name, "labeling")

    images_ext = config_dict["image_exstension"]

    replicate_dataset(classes,
                      dataset_path,
                      source_folder,
                      classes_file,
                      images_ext,
                      update_txts)
