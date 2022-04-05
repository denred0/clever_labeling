import torch
import numpy as np


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
    intersection = np.clip(x2 - x1, 0, x2 - x1) * np.clip(y2 - y1, 0, y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return round(intersection / (box1_area + box2_area - intersection + 1e-6), 2)


if __name__ == "__main__":
    boxes_preds = [50, 50, 100, 100]
    boxes_labels = [0, 0, 100, 100]
    print(intersection_over_union_box(boxes_preds, boxes_labels, "corners"))
