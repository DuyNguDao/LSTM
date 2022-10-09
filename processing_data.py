from glob import glob
import os
import shutil
import yaml

path_dataset = '/home/duyngu/Downloads/Dataset_Human_Action'
path_save = '/home/duyngu/Downloads/Dataset_Human_Action/dataset'
if os.path.exists(path_save):
    shutil.rmtree(path_save)
os.mkdir(path_save)
class_names = ['Standing', 'Walking', 'Sitting', 'Lying_Down', 'Stand_up', 'Sit_down', 'Fall_Down']
for name in class_names:
    path_class = os.path.join(path_save, name)
    if not os.path.exists(path_class):
        os.mkdir(path_class)
office = ['Office', 'Lecture_room', 'Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']
list_office = glob(path_dataset + '/*/*')
list_office = [i for i in list_office if list_office if i.split('/')[-1] == 'images' and i.split('/')[-2] in office]
print(list_office)
for path_img in list_office:
    list_name = os.listdir(path_img)
    for name in list_name:
        path_class = os.path.join(path_img, name)
        list_id_video = glob(path_class + '/*')
        save = os.path.join(path_save, name)
        for path_id in list_id_video:
            save1 = os.path.join(save, '%.3d'%(len(glob(save + '/*')) + 1))
            os.mkdir(save1)
            path_frame = glob(path_id + '/*.jpg')
            for frame in path_frame:
                shutil.copy(frame, save1)
            if len(glob(save1 + '/*.jpg')) < 31:
                shutil.rmtree(save1)
