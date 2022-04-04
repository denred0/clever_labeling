import os
import shutil
import argparse

from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder, recreate_folder


def merge_txts_labels(input_dir: str,
                      output_dir: str,
                      image_ext: str) -> None:
    #
    label_folders = os.listdir(input_dir)

    all_txts_paths = get_all_files_in_folder(input_dir, ["*.txt"])
    unique_txt_files = list(set([file.name for file in all_txts_paths]))

    for file in tqdm(unique_txt_files):
        result = []
        images = []

        for label_folder in label_folders:
            file_path = Path(input_dir).joinpath(label_folder).joinpath(file)
            if os.path.isfile(file_path):
                with open(file_path) as txt_file:
                    lines = [line.rstrip() for line in txt_file.readlines()]

                result.extend(lines)

                images.append(Path(input_dir).joinpath(label_folder).joinpath(file.split(".")[0] + "." + image_ext))

        with open(Path(output_dir).joinpath(file), 'w') as f:
            for line in result:
                f.write("%s\n" % str(line))

        shutil.copy(images[0], Path(output_dir))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('ext_images', type=str, default='jpg', help='Images_ext')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    #проверять дубли и пересекающиеся боксы больше, чем на 0.8 например
    # еще задать обязательный класс, который точно должен быть на каждом изображении и еще проверять на пустые изображения - вообще без классов
    # посчитать количество классов
    opt = parse_opt()

    project_name = opt.project_name
    # project_name = "door_smoke"

    ext_images = opt.ext_images
    # ext_images = "jpg"

    directories = next(os.walk(f"data/{project_name}/labeling"))[1]

    input_dir = "data/door_smoke/merge_labels/input"
    recreate_folder(input_dir)

    for dir in directories:
        shutil.copytree(f"data/{project_name}/labeling/{dir}/data", f"{input_dir}/{dir}")

    output_dir = "data/door_smoke/merge_labels/output"
    recreate_folder(output_dir)

    merge_txts_labels(input_dir, output_dir, ext_images)
