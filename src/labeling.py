import torch
import os
import cv2
import argparse
import yaml

from tqdm import tqdm

from my_utils import get_all_files_in_folder, get_last_exp_number


def pseudolabeling(data_dir: str,
                   weights: str,
                   threshold: float,
                   nms: float,
                   model_input_image_size: int,
                   images_ext: str,
                   count_of_images_to_markup: int) -> None:
    #
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
    model.conf = threshold
    model.iou = nms
    model.classes = [int(data_dir.split(os.sep)[-2].split("_")[0])]

    images = get_all_files_in_folder(data_dir, [f"*.{images_ext}"])
    images = [im.stem for im in images]
    txts = get_all_files_in_folder(data_dir, [f"*.txt"])
    txts = [txt.stem for txt in txts]

    images_for_labeling = sorted(list(set(images) - set(txts)))
    if count_of_images_to_markup == -1:
        images_for_labeling = [imm + "." + images_ext for imm in images_for_labeling]
    else:
        images_for_labeling = [imm + "." + images_ext for imm in images_for_labeling][:count_of_images_to_markup]

    labeled = 0
    for im in tqdm(images_for_labeling):
        img = cv2.imread(os.path.join(data_dir, im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        detections = model(img, size=model_input_image_size)
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
            labeled += 1
            with open(os.path.join(data_dir, im.split(".")[0] + ".txt"), 'w') as f:
                for item in detections_result:
                    f.write("%s\n" % (str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(
                        item[3]) + ' ' + str(item[4])))

    print(f"Labeled images: {labeled}/{len(images_for_labeling)}")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('class_for_training', type=str, help='Name of class for training')
    parser.add_argument('--weights', type=str, help='pretraining weights')
    parser.add_argument('--count_of_images_to_markup', '--img_count', type=int,
                        help='Number of images for labeling')
    parser.add_argument('--th', type=float)
    parser.add_argument('--nms', type=float)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    project_name = opt.project_name
    class_for_training = opt.class_for_training
    count_of_images_to_markup = opt.count_of_images_to_markup
    threshold = opt.th
    nms = opt.nms
    weights = opt.weights

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

    source_folder = f"data/{project_name}/labeling"

    images_ext = config_dict['image_exstension']
    source_folder_class = os.path.join(source_folder,
                                       str(classes.index(class_for_training)) + "_" + class_for_training)

    data_dir = os.path.join(source_folder_class, "data")
    if not weights:
        project_path = os.path.join(source_folder_class, "training", "runs")
        last_exp_number = get_last_exp_number(project_path)
        weights = os.path.join(source_folder_class, "training", "runs", "exp" + str(last_exp_number), "weights",
                               "best.pt")
    model_input_image_size = config_dict['model_input_image_size']
    if not threshold:
        threshold = config_dict['threshold']

    if not nms:
        nms = config_dict['nms']

    if not count_of_images_to_markup:
        count_of_images_to_markup = config_dict['count_of_images_to_markup']

    pseudolabeling(data_dir,
                   weights,
                   threshold,
                   nms,
                   model_input_image_size,
                   images_ext,
                   count_of_images_to_markup)
