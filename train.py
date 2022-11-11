"""
Dao Duy Ngu
"""
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import model deep learning
from models.rnn import RNN, GRUModel
from tqdm import tqdm
from collections import OrderedDict
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from Data_Loader.dataset import processing_data

logging.addLevelName(logging.WARNING, "")

# clear memory cuda
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# Get parameter
with open("./config.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# parameter
input_dataset_train = data['dataset-path-train']
input_dataset_test = data['dataset-path-test']
epochs = data['epochs']
batch_size = data['batch-size']
input_size = data['img-size']
num_frame = data['num-frame']
path_save_model = data['project']
features, labels = [], []
# Load dataset train
with open(input_dataset_train, 'rb') as f:
    fts, lbs = pickle.load(f)
    features.append(fts)
    labels.append(lbs)
del fts, lbs

# ****************************************** NORMALIZE CLASS ****************************************************
labels = np.concatenate(labels, axis=0).argmax(1)
# id = ((labels == 0) | (labels == 6))
# labels[id] = 0
# id = ((labels == 2) | (labels == 3) | (labels == 4))
# labels[id] = 2
# id = (labels == 5)
# labels[id] = 3
# id = (labels == 5)
# labels[id] = 0
# labels[~id] = 1
features = np.concatenate(features, axis=0)  # 30x34
features = features[:, :, :, :2]
# get 15 frame
features = features[:, ::2, :, :]
features = processing_data(features)
x_train = features
y_train = labels
print(" --------- Number class train---------")
for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

# load dataset test
features, labels = [], []
with open(input_dataset_test, 'rb') as f:
    fts, lbs = pickle.load(f)
    features.append(fts)
    labels.append(lbs)
del fts, lbs
# ****************************************** NORMALIZE CLASS ****************************************************
labels = np.concatenate(labels, axis=0).argmax(1)
# id = ((labels == 0) | (labels == 6))
# labels[id] = 0
# id = ((labels == 2) | (labels == 3) | (labels == 4))
# labels[id] = 2
# id = (labels == 5)
# labels[id] = 3
# id = (labels == 5)
# labels[id] = 0
# labels[~id] = 1
features = np.concatenate(features, axis=0) # 30x34
features = features[:, :, :, :2]
# get 15 frame
features = features[:, ::2, :, :]
features = processing_data(features)
x_valid = features
y_valid = labels
print(" --------- Number class test---------")
for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

del features, labels
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                              torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(x_valid, dtype=torch.float32),
                            torch.tensor(y_valid))

del x_train, x_valid, y_train, y_valid

# create folder save
if not os.path.exists(path_save_model):
    os.mkdir(path_save_model)
count = 0
# check path save
while os.path.exists(path_save_model + f'/exp{count}'):
    count += 1
# create new folder save
path_save_model = path_save_model + f'/exp{count}'
os.mkdir(path_save_model)

# load data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True,
    num_workers=batch_size, pin_memory=True)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=batch_size, pin_memory=True)

del train_dataset, val_dataset

classes_name = ['Sit down', 'Lying Down', 'Walking', 'Stand up', 'Standing', 'Fall Down', 'Sitting']
# classes_name = ['Fall Down', 'Other action']
# classes_name = ['Siting', 'Lying Down', 'Walking or Standing', 'Fall Down']
print("Class name:", classes_name)

# load model LSTM
model = RNN(input_size=26, num_classes=len(classes_name), device=device)
# model = GRUModel(input_dim=34, hidden_dim=256, layer_dim=1, output_dim=len(classes_name))
model = model.to(device)

# config function loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

data_loaders = {
    'train': train_loader,
    'validation': val_loader
}


def train_model(model, criterion, optimizer, num_epochs):
    """
    function: Training model
    :param model: model deep learning
    :param criterion: loss
    :param optimizer: optimizer
    :param num_epochs: number epochs
    :return:
    """
    best_loss_acc = -1
    loss_list = {'train': [], 'valid': []}
    acc_list = {'train': [], 'valid': []}

    for epoch in range(num_epochs):
        # train
        losses_train = 0.0
        train_corrects = 0
        last_time = time.time()
        model.train()
        pbar_train = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_vid, labels in pbar_train:
            batch_vid, labels = batch_vid.to(device), labels.to(device)
            outputs = model(batch_vid)
            loss = criterion(outputs, labels)
            # setup grad and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_train += loss.item()
            _, preds = torch.max(outputs, 1)
            train_corrects += (preds == labels.data).detach().cpu().numpy().mean()
            del batch_vid, labels
            # set memomy
            total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            pbar_train.set_postfix(OrderedDict({'Loss': loss.item(),
                                                'Memory': "%0.2f GB / %0.2f GB" % (used_memory_before / 1024,
                                                                                   total_memory / 1024)}))

        epoch_loss = losses_train / len(train_loader)
        loss_list['train'].append(epoch_loss)
        epoch_acc = train_corrects/len(train_loader)
        acc_list['train'].append(epoch_acc)
        logging.warning('Train: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc, epoch_loss,
                                                                         str(datetime.timedelta(seconds=time.time() - last_time))))
        # validation
        last_time = time.time()
        losses_val = 0.0
        val_corrects = 0
        model.eval()
        with torch.no_grad():
            for batch_vid, labels in val_loader:
                batch_vid, labels = batch_vid.to(device), labels.to(device)
                outputs = model(batch_vid)
                loss = criterion(outputs, labels)
                losses_val += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels.data).detach().cpu().numpy().mean()
                del batch_vid, labels

            epoch_loss = losses_val / len(val_loader)
            loss_list['valid'].append(epoch_loss)
            epoch_acc = val_corrects / len(val_loader)
            acc_list['valid'].append(epoch_acc)
            logging.warning('Validation: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc,
                                                                                          epoch_loss,
                                                                                          str(datetime.timedelta(seconds=time.time() - last_time))))

            if best_loss_acc == -1:
                best_loss_acc = epoch_acc
            if best_loss_acc <= epoch_acc:
                best_loss_acc = epoch_acc
                torch.save(model.state_dict(), path_save_model + '/best.pt')
                logging.warning('Saved best model at epoch {}'.format(epoch))

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(acc_list['train'], label="Train Accuracy")
        plt.plot(acc_list['valid'], label="Val Accuracy")
        plt.xlabel("epoch")
        plt.title("Accuracy")

        # plt.grid()
        plt.legend(loc="best")
        plt.subplot(1, 2, 2)
        plt.plot(loss_list['train'], label="Train Loss")
        plt.plot(loss_list['valid'], label="Val Loss")
        plt.xlabel("epoch")
        plt.title("Loss")
        plt.legend(loc="best")
        # plt.grid()
        fig.savefig(path_save_model + '/result.png', dpi=500)
        plt.close(fig)
        del fig

    return model


def main():
    """
    function: training model
    :return:
    """
    model_trained = train_model(model, criterion, optimizer, num_epochs=epochs)
    torch.save(model_trained.state_dict(), path_save_model + '/last.pt')
    logging.warning('Saved last model at {}'.format(path_save_model, "/last.pt"))
    print("Complete !")


if __name__ == '__main__':
    main()
