"""
Dao Duy Ngu
"""
import torch
from torchvision import datasets, transforms
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
from models.rnn import RNN
from tqdm import tqdm
from collections import OrderedDict
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import yaml

logging.addLevelName(logging.WARNING, "")

# clear memory cuda
torch.cuda.empty_cache()
# Get parameter
with open("./config.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# parameter
input_dataset = data['dataset-path']
epochs = data['epochs']
batch_size = data['batch-size']
input_size = data['img-size']
num_frame = data['num-frame']
path_save_model = data['project']

# Load dataset
features, labels = [], []
with open(input_dataset, 'rb') as f:
    fts, lbs = pickle.load(f)
    features.append(fts)
    labels.append(lbs)
del fts, lbs

features = np.concatenate(features, axis=0)  # 30x34
features = np.concatenate([features[:, :, 0:1, :], features[:, :, 5:, :]], axis=2)
features = features[:, :, :, :2].reshape(len(features), 30, 26)
labels = np.concatenate(labels, axis=0).argmax(1)

print(" --------- Number class before balance ---------")
for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

# # imbalance data
ids = np.array([[i] for i in range(len(features))])
# under = RandomUnderSampler(sampling_strategy='majority')
#
remove_min = (labels != 5)  # label == 3 or 6
label = labels[remove_min]
id = ids[remove_min]
label_min = labels[~remove_min]
id_min = ids[~remove_min]
id_min = id_min[:13000]
label_min = label_min[:13000]

# remove_min = (label != 1)  # label == 3 or 6
# label_min = np.concatenate((label[~remove_min], label_min), axis=0)
# id_min = np.concatenate((id[~remove_min], id_min), axis=0)
# label = label[remove_min]
# id = id[remove_min]
#
# for _ in range(4):
#     id, label = under.fit_resample(id, label)
#
ids = np.concatenate((id, id_min), axis=0)
labels = np.concatenate((label, label_min), axis=0)

print(" --------- Number class after balance ---------")

for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

# get feature
ids = [i[0] for i in ids]
features = features[ids]
# ---------------------------------------------------------


x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2,
                                                      random_state=9)
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                              torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(x_valid, dtype=torch.float32),
                            torch.tensor(y_valid))

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

classes_name = ['Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down']
print("Class name:", classes_name)

# load model LSTM
model = RNN(input_size=26, num_classes=len(classes_name), device=device)
model = model.to(device)

# config function loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-3)

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
    best_loss_val = -1
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
            # set memomy
            total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            pbar_train.set_postfix(OrderedDict({'Loss': loss.item(),
                                                'Memory': "%0.2f GB / %0.2f GB" % (used_memory_before / 1024,
                                                                                   total_memory / 1024)}))
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        epoch_loss = losses_train / len(train_dataset)
        loss_list['train'].append(epoch_loss)
        epoch_acc = train_corrects.double()/len(train_dataset)
        acc_list['train'].append(epoch_acc.cpu().numpy())
        logging.warning('Train: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc.cpu().numpy(), epoch_loss,
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
                val_corrects += torch.sum(preds == labels.data)

            epoch_loss = losses_val / len(val_dataset)
            loss_list['valid'].append(epoch_loss)
            epoch_acc = val_corrects.double()/len(val_dataset)
            acc_list['valid'].append(epoch_acc.cpu().numpy())
            logging.warning('Validation: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc.cpu().numpy(),
                                                                                          epoch_loss,
                                                                                          str(datetime.timedelta(seconds=time.time() - last_time))))
            if best_loss_val == -1:
                best_loss_val = losses_val
            if best_loss_val >= losses_val:
                best_loss_val = losses_val
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
