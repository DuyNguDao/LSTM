# HUMAN ACTION RECOGNITION WITH LSTM

## Introduction
This is project train human action recognition depend on skeleton with LSTM. Skeleton extracted with model YOLOv7-Pose

## Dev
```
Member:
- Dao Duy Ngu
- Le Van Thien
```
## Usage
### Install package
```
git clone https://github.com/DuyNguDao/LSTM.git
```
```
cd LSTM
```
```
conda create --name human_action python=3.8
```
```
pip install -r requirements.txt
```
## Dataset
This is dataset include with 7 class: sitting, sit down, standing, walking, stand up, lying down, fall down
## video
[video]()
## pose
### pose using YOLOv7-Pose
[pose yolov3]()
### pose using YOLOv3 + Alphapose
[pose yolov7]()

## Training
```commandline
python train.py
```
## Test
```commandline
python test.py
```
