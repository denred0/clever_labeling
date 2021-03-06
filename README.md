# Clever labeling
**Clever labeling (CL)** helps to create labeling for object detection task. 

CL tested together with Yolo_mark (https://github.com/AlexeyAB/Yolo_mark) for labeling. 


## How it works
1. You start labeling your data with Yolo_mark.
2. CL trains YOLOv5 on your labeled data so far. 
3. If mAP@:.5:.95 is more than 0.8 (you can set your value) after training then CL creates bboxes for images that you did't label before the current moment.
4. If mAP@:.5:.95 is less than 0.8 than CL will wait for 20 minutes (you can set your value) and try again with data that you labeled so far. 


## Install
```python
git clone https://github.com/denred0/clever_labeling.git
cd clever_labeling
pip install -r requirements.txt
```

## Data Preparation
1. Create a folder "data". Create a folder with name of your project inside "data" folder. For example, "animals_detection". 
2. Create file with list of classes "classes.txt" inside your project folder. 
<br>Example of structure of "classes.txt" file:
<br>dog
<br>cat
<br>pig
<br>Also you can see example of "classes.txt" in data/sample_project
3. Create folder "dataset" inside your project folder and copy images for labeling to "dataset" folder. 
4. As a result your catalog should be like this:
<br>clever_labeling
<br>├── data
<br>│   ├── animals_detection
<br>│   │   ├── classes.txt
<br>│   │   ├── dataset
<br>│   │   │   ├── image1.jpg
<br>│   │   │   ├── image2.jpg
<br>│   │   │   ├── image3.jpg
5. Copy "labeling_config.yaml" from data/sample_project to folder with your project. You can configure training and pseudolabeling of your project using this "labeling_config.yaml".

## Prepare dataset
Run script `prepare_dataset.py`
```python
python src/prepare_dataset.py %project_folder_name% 
python src/prepare_dataset.py animals_detection 
```

It will create folder "labeling" with subfolder for every class.<br>You will labeling every class separately. I noticed that it is more precise and convenient. 
Subfolder for every class will have only one class for labeling with index 0. When you markup all classes you can merge all txts together and every class will have own index according "classes.txt".
You can find the merging process in **Merging results** part of this tutorial. 

**prepare_dataset.py** has additional parameter _--upd_txts_ that means that you want to create txts for every class and fill them with values from data/animals_detection/dataset. But be careful It rewrites txts for classes if they existed before. 

## Training
To start training run:
```python 
python src/train.py %project_folder_name% %class_name%
python src/train.py animals_detection dog
```

This script creates a folder "animals_detection/labeling/dog/**training**" and all training results are saved to this folder.

**Additional parameters:**
<br>_--resume_weights_ - path to weights to resume training

You can change params in **labeling_config.yaml** between training attempts. 


## Pseudo labeling
To start training run:
```python 
python src/labeling.py %project_folder_name% %class_name%
python src/labeling.py animals_detection dog
```

**Additional parameters:**
<br>_--count_of_images_to_markup_ - count of images to markup if mAP will be greater than min mAP. 
<br>_--th_ - min threshold for label sample. 
<br>_--nms_ - nonmax suspression for detection
<br>_--exp_ - use weights of specific experiment

It will markup _count_of_images_to_markup_ that you didn't markup so far.  

If you don't like results of labeling you can run **src/train.py** again. In this case **src/train.py** takes all data that you markup so far.  

## Merging results

You labeled all classes separately and can merge results. 
<br>Run script `merge_labels.py`
```python
python src/merge_labels.py %project_folder_name% 
python src/merge_labels.py animals_detection
```

This script creates a folder _animals_detection/merge_ with resulting markup. 

Additionally you can check your dataset for collision. 
Check **sample_project/merge_config.yaml** to configure merging process:
<br>_classes_to_merge_ - classes that you want to merge. You can set [] - it means that all classes should be merged.
<br>_obligatory_classes_ - check that every image has markup for specific class. 
<br>_iou_ - check that bboxes doesn't have overlapping more that this value. 

**src/merge_labels.py** creates folder for every collision case: 1_high_iou, 2_without_obligatory_classes, 3_empty_images (images without any bbox).
<br>You can fix markup in every that folder consequentially and then start script **src/update_and_merge.py %project_name% %upd%**
<br>_upd_ parameter means type of update. Possible values: _iou_, _obl_ (for obligatory_classes), _emp_ (for empty images). 

For example, you fixed markup inside folder _1_high_iou_, It means that you should update markup and resulting dataset and run the script:
```python
python src/update_and_merge.py animals_detection iou
```

Now you can fix markup in _2_without_obligatory_classes_ folder and etc. 