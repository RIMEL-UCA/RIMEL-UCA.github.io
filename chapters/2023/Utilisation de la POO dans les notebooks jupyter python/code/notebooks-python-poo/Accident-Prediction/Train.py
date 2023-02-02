import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import time
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_targets_tensor(file_path):
    return torch.from_numpy(np.loadtxt(file_path, dtype = np.float32)).to(device = device)

def load_dataset(dataset_paths):
    lst = []
    print('generating features for all video clips')
    for dataset_path in dataset_paths:
        video_clip = np.loadtxt(dataset_path[0], dtype = np.float32)
        feature_tensors = torch.from_numpy(video_clip).to(device = device)

        targets = get_targets_tensor(dataset_path[1])
        # TODO: make sure feature_tensors.device == cuda

        lst.append((feature_tensors, targets))
        print(len(lst))
    print('done generating features for all video clips', len(lst))
    return lst


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_dataset_path():
    #  TODO: Need to change this to load the actual dataset
    lst = []
    for _ in range(1, 86):
        video_path = f'drive/My Drive/dataset/train/videoclips/clip_' \
                     f'{_}/feature_tensors.txt'
        target_path = f'drive/My Drive/dataset/train/groundtruth/clip_{_}.txt'
        lst.append((video_path, target_path))
    return lst

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden)).to(device = device)
        x = self.input_layer(combined)
        hidden = self.hidden_layer_1(x)
        output = self.output_layer(hidden)
        sigmoid_op = self.sigmoid(output)
        return sigmoid_op, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size).to(device = device)

# TODO: add Exponential loss, and move it to CUDA
def _train(video_sequence_tensor, true_value_tensor, rnn, criterion, optimizer):
    """
    This function scope is over a video clip,
    its supposed to get a video frame tensor generated using Genaret_feature
    :param true_value_tensor:
    :param video_sequence_tensor: [n, 4096], where n is the length of video
    sequence
    :return:
    """
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    prediction_list = []
    current_loss = 0
    for i in range(video_sequence_tensor.size()[0]):
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
        prediction_list.append(prediction_tensor)
    # get prediction for every frame
    prediction_tensor = torch.cat(prediction_list).to(device = device)

    loss = criterion(prediction_tensor, true_value_tensor)
    # we want Exponential Loss here
    loss.backward()  # backpropogate

    optimizer.step()

    return loss.item()  # return  total loss for the current video sequence
def train(video_clip_target):

    rnn = RNN(4096, n_hidden).to(device = device)

    # TODO: Change this later to Exponential Loss
    criterion = nn.BCELoss()

    optimizer = optim.SGD(rnn.parameters(), lr=0.001, momentum=0.9)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        random.shuffle(video_clip_target)  # random the video clips (so the model does not
        # memorize anything
       
        for idx, data_item in enumerate(video_clip_target):
          
            feature_tensors = data_item[0]
            targets = data_item[1]
            
            loss = _train(feature_tensors, targets, rnn , criterion, optimizer)
            current_loss += loss

        all_losses.append(current_loss)
        print('epochs=', epoch, 'total Loss in this epoch=', current_loss,
              'time since start=', timeSince(start))
        current_loss = 0
        # Save the model
        torch.save(rnn.state_dict(), 'rnn_optimized' + str(
                epoch) +
                   '.model')

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    plt.savefig('total_loss.png')

n_hidden = 2048
epochs = 40
train(video_clip_target)
del dataset[83]
del video_clip_target[83]
dataset = get_dataset_path()
video_clip_target = load_dataset(dataset)
del dataset[28]
del video_clip_target[28]


def _test(video_sequence_tensor, rnn):
    """
    This function scope is over a video clip,
    its supposed to get a video frame tensor generated using Genaret_feature
    :param true_value_tensor:
    :param video_sequence_tensor: [n, 4096], where n is the length of video
    sequence
    :return:
    """
    hidden = rnn.initHidden()
    prediction_list = []
    for i in range(video_sequence_tensor.size()[0]):
        # for ith frame in the video frame
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
        prediction_list.append(prediction_tensor)
        # get prediction for every frame
    prediction_tensor = torch.cat(prediction_list)
    # get prediction for every frame


    return prediction_tensor.cpu().data.numpy()  # return  total loss for the current video sequence
def test(video_clip_target):
    scores = []
    rnn = RNN(4096, n_hidden).to(device=device)
    rnn.load_state_dict(torch.load(
            'rnn_optimized40.model'))
    rnn.eval()
    yhat = []
    y = []
    for data_item in video_clip_target:
        feature_tensors = data_item[0]
        targets = data_item[1].cpu().data.numpy().astype(int)
        # print(targets)
        output = _test(feature_tensors, rnn)
        scores.append(np.array(output))
        output[output >= 0.26] = 1
        output[output < 0.26] = 0
        output = output.astype(int)
        # print(output)
        # print('current clip accuracy = ', np.mean(output == targets))
        # print(classification_report(targets, output))
        yhat.append(output)
        y.append(targets)

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    print(classification_report( y, yhat ))
    print(confusion_matrix(y, yhat))
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    print('tn = ', tn ,' fp= ', fp, 'fn = ',fn,' tp = ',tp)
    return scores, y
scores , true = test(video_clip_target)
scores = np.concatenate(scores)
fpr, tpr, thresholds = metrics.roc_curve(true, scores)
roc_auc = metrics.auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(optimal_threshold)
roc_auc
plt.plot(fpr, tpr)
plt.show()
