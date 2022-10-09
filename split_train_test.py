from glob import glob
import os
import shutil
import random

path_dataset = '/home/duyngu/Downloads/Dataset_Human_Action/dataset'
path_save = '/home/duyngu/Downloads/Dataset_Human_Action/dataset_split'

rate_train = 0.8

# create folder save
if os.path.exists(path_save):
    shutil.rmtree(path_save)
os.mkdir(path_save)

# create folder train and val
list_train_val = ['train', 'val']
for name in list_train_val:
    path_name = os.path.join(path_save, name)
    if os.path.exists(path_name):
        shutil.rmtree(path_name)
    os.mkdir(path_name)

# get class_name
class_name = os.listdir(path_dataset)

# create class name save

for name in class_name:
    path_name = os.path.join(path_dataset, name)
    list_id_class = glob(path_name + '/*')
    random.shuffle(list_id_class)
    path_save_train = os.path.join(path_save, list_train_val[0])
    path_save_train = os.path.join(path_save_train, name)
    if not os.path.exists(path_save_train):
        os.mkdir(path_save_train)
    for data_train in list_id_class[:int(len(list_id_class)*rate_train)]:
        shutil.copytree(data_train, os.path.join(path_save_train, data_train.split('/')[-1]))
    path_save_val = os.path.join(path_save, list_train_val[1])
    path_save_val = os.path.join(path_save_val, name)
    for data_train in list_id_class[int(len(list_id_class)*rate_train):]:
        shutil.copytree(data_train, os.path.join(path_save_val, data_train.split('/')[-1]))

