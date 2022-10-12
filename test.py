import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
# import model deep learning
from models.rnn import RNN
import numpy as np
from sklearn import metrics
from util.plot import plot_cm
from tqdm import tqdm
from Data_Loader.dataset import processing_data


def detect_image(path_test, path_model, batch_size=256):
    """
    function: detect face mask of folder image
    :param path_image: path of folder contain image
    :return: None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    class_names = ('Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down')
    model = RNN(input_size=30, num_classes=len(class_names), device=device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device=device)
    model.eval()

    # load dataset
    # Load dataset
    features, labels = [], []
    with open(path_test, 'rb') as f:
        fts, lbs = pickle.load(f)
        features.append(fts)
        labels.append(lbs)
    del fts, lbs

    features = np.concatenate(features, axis=0)  # 30x34
    features = processing_data(features)
    # features = np.concatenate([features[:, :, 0:1, :], features[:, :, 5:, :]], axis=2)
    # features = features[:, ::2, :, :]
    # features = features[:, :, :, :2].reshape(len(features), features.shape[1], features.shape[2]*features.shape[3])
    labels = np.concatenate(labels, axis=0).argmax(1)
    test_dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                                torch.tensor(labels))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=batch_size, pin_memory=True)
    truth = []
    pred = []
    pbar_test = tqdm(test_loader, desc=f'Evaluate', unit='batch')
    for batch_vid, labels in pbar_test:
        batch_vid, labels = batch_vid.to(device), labels.to(device)
        outputs = model(batch_vid)
        _, preds = torch.max(outputs, 1)
        truth.extend(labels.data.tolist())
        pred.extend(preds.tolist())
    CM = metrics.confusion_matrix(truth, pred).T
    precision = metrics.precision_score(truth, pred, average=None)
    recall = metrics.recall_score(truth, pred, average=None)
    accuracy = metrics.accuracy_score(truth, pred, normalize=True)
    f1_score = metrics.f1_score(truth, pred, average=None)
    print("Accuracy: ", round(accuracy, 2) * 100)
    for i in range(len(class_names)):
        print('****Precision-Recall-F1-Score of class {}****'.format(class_names[i]))
        print('Precision: ', precision[i])
        print('Recall: ', recall[i])
        print('F1-score', f1_score[i])
    with open('info_lstm/info_lstm.txt', 'w') as file:
        file.write('{} {} {}'.format(precision, recall, f1_score))
    plot_cm(CM, normalize=False, save_dir='info_lstm', names_x=class_names,
            names_y=class_names, show=False)
    print('Finishing!.')


if __name__ == '__main__':
    path_model = 'runs/exp2/best.pt'
    path_frame = '/home/duyngu/Downloads/Dataset_Human_Action/test_no_scale.pkl'
    detect_image(path_frame, path_model, batch_size=256)
