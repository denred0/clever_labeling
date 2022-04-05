import os
import argparse
from collections import defaultdict

from tqdm import tqdm

from my_utils import get_all_files_in_folder, read_config
from merge_labels import merge_txts_labels


def update_and_merge(project_name: str,
                     upd_type: str):
    #
    if upd_type == "iou":
        dir = os.path.join("data", project_name, "merge", "1_big_iou")
    elif upd_type == "obl":
        dir = os.path.join("data", project_name, "merge", "2_without_obligatory_classes")
    elif upd_type == "emp":
        dir = os.path.join("data", project_name, "merge", "3_empty_images")
    else:
        print(f"upd {upd_type} is not supported")
        return

    labeling_config = read_config(os.path.join("data", project_name, "labeling_config.yaml"))

    classes_file = labeling_config['classes_file']
    classes_file_path = os.path.join("data", project_name, classes_file)
    with open(classes_file_path) as file:
        classes = {k: v for (k, v) in enumerate([line.rstrip() for line in file])}

    txts = get_all_files_in_folder(dir, ["*.txt"])

    for txt in tqdm(txts, desc="Updating txts"):
        with open(txt) as txt_file:
            lines = [line.rstrip() for line in txt_file.readlines()]

        if txt.stem == "13_00307_0":
            print()

        # recreate txt for every class
        for cl in classes:
            open(os.path.join("data", project_name, "labeling", str(cl) + "_" + classes[cl], "data", txt.name), 'w')

        txt_dict = defaultdict(list)
        for line in lines:
            txt_dict[int(line.split()[0])].append(line)

        # fill txt with new values
        for cl, val in txt_dict.items():
            with open(os.path.join("data", project_name, "labeling", str(cl) + "_" + classes[cl], "data", txt.name),
                      'w') as f:
                for line in val:
                    f.write("%s\n" % str(line))

    merge_txts_labels(project_name)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('upd', type=str, help='Type of update')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    upd_type = opt.upd

    update_and_merge(project_name, upd_type)
