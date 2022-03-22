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
      - 

### Spliting data
run 