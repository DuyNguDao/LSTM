from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
import os
from PIL import Image
import random
import torch
import csv


class VideoFrameDataset(Dataset):
    def __init__(self, root, img_size=64, num_frame=30):
        self.classes_name = os.listdir(root)
        self.img_size = img_size
        self.num_frame = num_frame
        self.videos = []
        self.targets = []
        for name in self.classes_name:
            path_class = os.path.join(root, name)
            name_video = os.listdir(path_class)
            for id in name_video:
                self.videos.append(os.path.join(path_class, id))
                self.targets.append(name)

        self.transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video = torch.stack(self.read_video(video))
        label = self.targets[idx]
        label = self.classes_name.index(label)
        return video, label

    def read_video(self, path):
        list_frame = os.listdir(path)
        list_frame.sort()
        frames = []
        value_random = random.randrange(0, len(list_frame) - self.num_frame, 2)
        for idx, name in enumerate(list_frame):
            if idx >= value_random + self.num_frame or idx < value_random:
                continue
            image = Image.open(os.path.join(path, name))
            image_trans = self.transform(image)
            frames.append(image_trans)
        return frames


class PoseFrameDataset(Dataset):
    def __init__(self, root, size_pose=34, num_frame=30):
        self.classes_name = os.listdir(root)
        self.size_pose = size_pose
        self.num_frame = num_frame
        self.videos = []
        self.targets = []
        for name in self.classes_name:
            path_class = os.path.join(root, name)
            name_video = glob(path_class + '/*.csv')
            for id in name_video:
                file = open(id)
                data = list(csv.reader(file))[1:]
                file.close()
                if len(data) < self.num_frame:
                    continue
                self.videos.append(id)
                self.targets.append(name)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video = torch.stack(self.read_video(video))
        label = self.targets[idx]
        label = self.classes_name.index(label)
        return video, label

    def read_video(self, path):
        file = open(path)
        data = list(csv.reader(file))[1:]
        file.close()
        list_frame = [list(map(float, i)) for i in data]
        # list_frame.sort()
        frames = []
        if len(list_frame) < self.num_frame + 2:
            value_random = 0
        else:
            value_random = random.randrange(0, len(list_frame) - self.num_frame, 2)
        for idx, name in enumerate(list_frame):
            if idx >= value_random + self.num_frame or idx < value_random:
                continue
            # image = Image.open(os.path.join(path, name))
            image = name
            image_trans = torch.tensor(image)
            frames.append(image_trans)
        return frames





