import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from PIL import Image

# flag to true if you are drawing confusion matrix and sample images
drawing_cm = False
# preset for NN
image_scale = 36
epochs = 50
batch_size = 32
patience = 20
learning_rate = 0.05
# output string
output = ''
# check if there is a GPU present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
output += 'Using device: {}\n'.format(device)
labels = os.listdir('hw6_data/train')[1:]
# transform of the images for dataloader of pytorch
transformer = transforms.Compose([
    transforms.RandomResizedCrop(image_scale, ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.176, 0.172, 0.168], [0.166, 0.166, 0.165])
    ])
print('Loading image rescaled to {}'.format(image_scale))
output += 'Loading image rescaled to {}\n'.format(image_scale)


# custom image set dataset class
class ImageSets(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        # load images
        for label in os.listdir(root):
            path = os.path.join(root, label)
            if os.path.isdir(path):
                for r, _, fs in os.walk(path):
                    for f in fs:
                        if f.endswith('JPEG'):
                            full_path = os.path.join(r, f)
                            current_label = labels.index(label)
                            self.samples.append((full_path, current_label))
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    # my getitem also return the image's path for later evaluation and sampling
    def __getitem__(self, index):
        path, label = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label, path


# load train set, test set from image folders, also split the train set into train set and validation set by a portion
def load_split_train_test(train_dir, test_dir, train_portion=0.8):
    # train_data = datasets.ImageFolder(train_dir, transform=transformer)
    # test_data = datasets.ImageFolder(test_dir, transform=transformer)
    train_data = ImageSets(train_dir, transform=transformer)
    test_data = ImageSets(test_dir, transform=transformer)
    # calculate the split index
    n_train_examples = int(len(train_data) * train_portion)
    n_valid_examples = len(train_data) - n_train_examples
    # split train into train and validation
    splitted_train_data, splitted_valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])
    # data loaders
    trainloader = torch.utils.data.DataLoader(splitted_train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(splitted_valid_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader


# model evaluation
def model_eval(model, dataloader, criterion):
    eval_epoch_loss = 0
    eval_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for x, y, path in dataloader:
            x = x.to(device)
            y = y.to(device)
            fx = model(x)
            eval_loss = criterion(fx, y)
            eval_preds = torch.max(fx, 1)[1]
            eval_correct = eval_preds.eq(y.view_as(eval_preds)).sum()
            eval_acc = eval_correct.float() / eval_preds.shape[0]
            eval_epoch_loss += eval_loss.item()
            eval_epoch_acc += eval_acc.item()
    # calculate accuracy and loss
    eval_epoch_acc /= len(dataloader)
    eval_epoch_loss /= len(dataloader)

    return eval_epoch_acc, eval_epoch_loss


# helper function for plotting confusion matrix, this is a standard function from sklearn documentation
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# function for generating confusion matrix and image samples of results
def generate_confusion_matrix(model, pt_path, dataloader):
    model.to(device)
    model.load_state_dict(torch.load(pt_path))
    model.eval()
    preds = []
    trues = []
    # create containers for later storage of correct classified sample and incorrect classified samples
    bad_pred = dict()
    good_pred = dict()
    bad_pred[0] = []
    bad_pred[1] = []
    bad_pred[2] = []
    bad_pred[3] = []
    bad_pred[4] = []
    good_pred[0] = []
    good_pred[1] = []
    good_pred[2] = []
    good_pred[3] = []
    good_pred[4] = []
    # loop through data
    with torch.no_grad():
        for x, y, paths in dataloader:
            x = x.to(device)
            fx = model(x)
            pred = torch.max(fx, 1)[1]
            preds += pred.cpu().numpy().tolist()
            trues += y.numpy().tolist()
            # get current batch predictions
            current_preds = pred.cpu().numpy().tolist()
            current_trues = y.numpy().tolist()
            # save image path for later
            for i in range(len(current_preds)):
                if current_preds[i] != current_trues[i]:
                    bad_pred[current_preds[i]].append(paths[i])
                else:
                    good_pred[current_preds[i]].append(paths[i])

    plot_confusion_matrix(trues, preds, classes=labels, normalize=True)
    plt.savefig('torch_cm.jpg')
    # save sampled images
    os.mkdir('torch_result_samples')
    for label in labels:
        os.mkdir('torch_result_samples/{}'.format(label))
        os.mkdir('torch_result_samples/{}/good'.format(label))
        os.mkdir('torch_result_samples/{}/bad'.format(label))
    # loop through good classification samples and bad classification samples
    for i in range(len(labels)):
        good_counter = 0
        bad_counter = 0
        for a in good_pred[i]:
            img = cv2.imread(a)
            tokens = a.split('\\')
            cv2.imwrite('torch_result_samples/{}/good/{}_{}.jpg'.format(labels[i], good_counter, tokens[-1][:-5]), img)
            good_counter += 1
            if good_counter == 100:
                break
        for b in bad_pred[i]:
            img = cv2.imread(b)
            tokens = b.split('\\')
            cv2.imwrite('torch_result_samples/{}/bad/{}_{}.jpg'.format(labels[i], bad_counter, tokens[-1][:-5]), img)
            bad_counter += 1
            if bad_counter == 100:
                break


# fully connected NN
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(3*image_scale*image_scale, 1000)
        self.fc2 = nn.Linear(1000, 750)
        self.fc3 = nn.Linear(750, 500)
        self.fc4 = nn.Linear(500, 250)
        self.fc5 = nn.Linear(250, 125)
        self.output_fc = nn.Linear(125, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.output_fc(x)
        return x


# complex fc net
class FCNet_complex(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(3*image_scale*image_scale, 1000)
        self.fc2 = nn.Linear(1000, 900)
        self.fc3 = nn.Linear(900, 800)
        self.fc4 = nn.Linear(800, 700)
        self.fc5 = nn.Linear(700, 600)
        self.fc6 = nn.Linear(600, 500)
        self.fc7 = nn.Linear(500, 400)
        self.fc8 = nn.Linear(400, 300)
        self.output_fc = nn.Linear(300, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.output_fc(x)
        return x

# simple conv net
class FCNet_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(3*image_scale*image_scale, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 125)
        self.output_fc = nn.Linear(125, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_fc(x)
        return x


# 2d convolution NN
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(in_features=18 * 18 * 24, out_features=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        # convert tensors to shape for linear layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# complex conv net
class ConvNet_complex(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.fc = nn.Linear(in_features=9 * 9 * 24, out_features=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        # convert tensors to shape for linear layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# simple structure conv net
class ConvNet_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(in_features=36 * 36 * 12, out_features=5)
        # self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # convert tensors to shape for linear layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.sm(x)
        return x


if __name__ == '__main__':
    # prepare the loader
    trainloader, validloader, testloader = load_split_train_test('hw6_data/train', 'hw6_data/test', train_portion=0.8)
    if not drawing_cm:
        print('Size of training set: {}'.format(len(trainloader.dataset)))
        print('Size of validation set: {}'.format(len(validloader.dataset)))
        print('Size of testing set: {}'.format(len(testloader.dataset)))
        output += 'Size of training set: {}\n'.format(len(trainloader.dataset))
        output += 'Size of validation set: {}\n'.format(len(validloader.dataset))
        output += 'Size of testing set: {}\n'.format(len(testloader.dataset))

        # create the model
        # model = FCNet()
        # model = FCNet_simple()
        # model = FCNet_complex()

        # model = ConvNet()
        model = ConvNet_simple()
        # model = ConvNet_complex()

        # print model structure
        print(model)
        output += str(model)

        # get model parameters
        params = list(model.parameters())
        print('Model has {} params'.format(len(params)))
        output += 'Model has {} params\n'.format(len(params))

        # model optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # use the 'device' to process the model
        model = model.to(device)
        criterion = criterion.to(device)
        print('Using SGD optimizer, Loss: CrossEntropyLoss, learning rate set at : {}'.format(learning_rate))
        output += 'Using SGD optimizer, Loss: CrossEntropyLoss, learning rate set at : {}\n'.format(learning_rate)

        # start the training process
        print('Training started with {} scheduled epochs, batch size {}'.format(epochs, batch_size))
        output += 'Training started with {} scheduled epochs, batch size {}\n'.format(epochs, batch_size)
        train_accs = []
        train_losses = []
        eval_accs = []
        eval_losses = []
        best_loss = 999999999
        patience_counter = 0
        # do all the epochs
        for epoch in range(epochs):
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for x, y, path in trainloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                fx = model(x)
                train_loss = criterion(fx, y)
                train_preds = fx.max(1, keepdim=True)[1]
                correct = train_preds.eq(y.view_as(train_preds)).sum()
                train_acc = correct.float() / train_preds.shape[0]
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            train_epoch_acc /= len(trainloader)
            train_epoch_loss /= len(trainloader)

            # evaluate the model with validation set
            eval_epoch_acc, eval_epoch_loss = model_eval(model, validloader, criterion)

            # save the best model
            if eval_epoch_loss < best_loss:
                best_loss = eval_epoch_loss
                torch.save(model.state_dict(), 'bestFC-model.pt')

            # increment the patience counter if loss does not decrease
            # or else, reset the counter
            if eval_epoch_loss > best_loss:
                patience_counter += 1
            else:
                patience_counter = 0

            # early stop the training is the loss does not decrease for too long
            if patience_counter == patience:
                print('Epoch: {} Train Loss: {} Train Acc: {} Valid Loss: {} Valid Acc: {}'
                      .format(epoch, train_epoch_loss, train_epoch_acc, eval_epoch_loss, eval_epoch_acc))
                print('EARLY STOPPED AT EPOCH: {} PATIENCE: {}\n'.format(epoch, patience))
                output += 'Epoch: {} Train Loss: {} Train Acc: {} Valid Loss: {} Valid Acc: {}\n'\
                    .format(epoch, train_epoch_loss, train_epoch_acc, eval_epoch_loss, eval_epoch_acc)
                output += 'EARLY STOPPED AT EPOCH: {} PATIENCE: {}\n'.format(epoch, patience)
                break
            else:
                print('Epoch: {} Train Loss: {} Train Acc: {} Valid Loss: {} Valid Acc: {}'
                    .format(epoch, train_epoch_loss, train_epoch_acc, eval_epoch_loss, eval_epoch_acc))
                output += 'Epoch: {} Train Loss: {} Train Acc: {} Valid Loss: {} Valid Acc: {}\n'\
                    .format(epoch, train_epoch_loss, train_epoch_acc, eval_epoch_loss, eval_epoch_acc)

            # store the accuracy history and loss history
            train_accs.append(train_epoch_acc)
            train_losses.append(train_epoch_loss)
            eval_accs.append(eval_epoch_acc)
            eval_losses.append(eval_epoch_loss)

        # plot the accuracy history and loss history
        plt.clf()
        plt.plot(train_accs, label='Training Acc')
        plt.plot(eval_accs, label='Validation Acc')
        plt.legend(frameon=False)
        plt.savefig('Accs.jpg')

        plt.clf()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(eval_losses, label='Validation Loss')
        plt.legend(frameon=False)
        plt.savefig('Loss.jpg')

        # evaluate the model with test set
        test_epoch_acc, test_epoch_loss = model_eval(model, testloader, criterion)

        print('Test Loss: {} Test Acc: {}'.format(test_epoch_loss, test_epoch_acc))
        output += 'Test Loss: {} Test Acc: {}\n'.format(test_epoch_loss, test_epoch_acc)

        # write all output to log
        log = open('log.txt', '+w')
        log.write(output)
        log.close()
    else:
        model = ConvNet()
        generate_confusion_matrix(model, 'Conv/bestconv-model.pt', testloader)


