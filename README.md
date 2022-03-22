# Clever labeling
**Clever labeling (CL)** helps create labeling for object detection task. 

CL tested together with Yolo_mark (https://github.com/AlexeyAB/Yolo_mark) for labeling. 


## How it works
1. You start labeling your data with Yolo_mark.
2. CL train YOLOv5 on your labeled data so far. 
3. If mAP@:.5:.95 is more than 0.8 (you can set your value) after training then CL create bboxes for images that you did't label before the current moment.
4. If mAP@:.5:.95 is less than 0.8 than CL will wait for 20 minutes (you can set your value) and try again with data that you labeled so far. 


## Install
```python
git clone https://github.com/denred0/clever_labeling.git
cd clever_labeling
pip install -r requirements.txt
```

### Data Preparation
1. Create folder with name of your project. For example, "animals_detection". 
2. Create file with list of classes "classes.txt" inside your project folder. 
<br>Example of structure of "classes.txt" file:
<br>dog
<br>cat
<br>pig
3. Create folder "dataset" inside your project folder and copy images for labeling to "dataset" folder. 
4. In result your catalog should be like this:
<br>animals_detection
<br>├── classes.txt
<br>├── dataset
<br>│   ├── image1.jpg
<br>│   ├── image2.jpg
<br>│   ├── image3.jpg

### Spliting data
Run script `prepare_dataset.py`
```python
python src/prepare_dataset.py %project_folder_name% %classes file name%

python src/prepare_dataset.py animals_detection classes.txt
```

It will create folder labeling with subfolder for every class.<br>You will labeling every class separately. I noticed that it is more precisely and convenient. 

### Pseudo labeling
Run script `pseudo_labeling.py`
<br>**General parameters**:
<br>_project_name_ (required) 
<br>_classes_file_ (required)
<br>_images_ext_ (required)
<br>_class_for_training_ (required)
<br>
<br> **Train parameters**:
<br>_--weights_ (optional) default = yolov5_weights/yolov5m.pt
<br>_--min_samples_count_ (optional) default = 200. Min number of samples to start training.
<br>_--image_size_ (optional) default = 640
<br>_--batch_size_ (optional) default = 16
<br>_--epochs_ (optional) default = 250
<br>_--test_split_part_ (optional) default = 0.2
<br>_--min_map_ (optional) default = 0.8. Min mAP@:.5:.95 for pseudolabeling.
<br>_--sleep_training_min_ (optional) default = 20. Wait before next training attempt (min).
<br>_--max_attempts_ (optional) default = 10. Max number of attempts.
<br>
<br>**Inference params**:
<br>_--threshold_ (optional) default = 0.9
<br>_--nms_ (optional) default = 0.5

Example:
```python
python src/pseudo_labeling.py animals_detection classes.txt jpg dog --weights yolov5_weights/yolov5m.pt --test_split_part 0.15 --nms 0.6
```

