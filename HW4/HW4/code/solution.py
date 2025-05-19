import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle, tqdm, os
import time


def load_data(data_dir):
    '''
    To load the Cifar-10 Dataset from files and reshape the 
    images arrays from shape [3072,] to shape [3, 32, 32].

    Please follow the instruction on how to load the data and 
    labels at https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        data_dir: String. The directory where data batches are 
            stored.

    Returns:
        x_train: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.uint8)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int64)
        x_test: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.uint8)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int64)
    '''

    ### YOUR CODE HERE
    # Initialize arrays for training data
    x_train = []
    y_train = []

    # Load the 5 training batches
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        if i == 1:
            x_train = batch[b'data']
            y_train = batch[b'labels']
        else:
            x_train = np.vstack((x_train, batch[b'data']))
            y_train = np.concatenate((y_train, batch[b'labels']))

    # Load test data
    test_file_path = os.path.join(data_dir, 'test_batch')
    with open(test_file_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')

    x_test = test_batch[b'data']
    y_test = test_batch[b'labels']

    # Reshape from [N, 3072] to [N, 3, 32, 32]
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)

    # Convert labels to int64
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    ### END YOUR CODE

    return x_train, y_train, x_test, y_test


def preprocess(train_images, test_images, normalize=False):
    '''
    To preprocess the data by 
        (1).Rescaling the pixels from integers in [0,255) to 
            floats in [0,1), or 
        (2).Normalizing each image using its mean and variance. 

    Args:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.uint8)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.uint8)
        normalize: Boolean. To control to rescale or normalize 
            the images.

    Returns:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.float64)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.float64)
    '''
    ### YOUR CODE HERE
    # Convert to float64
    train_images = train_images.astype(np.float64)
    test_images = test_images.astype(np.float64)

    if normalize:
        # Normalize each image using its mean and variance
        for i in range(train_images.shape[0]):
            for c in range(3):  # For each color channel
                mean = np.mean(train_images[i, c, :, :])
                std = np.std(train_images[i, c, :, :])
                if std > 0:  # Avoid division by zero
                    train_images[i, c, :, :] = (train_images[i, c, :, :] - mean) / std

        for i in range(test_images.shape[0]):
            for c in range(3):
                mean = np.mean(test_images[i, c, :, :])
                std = np.std(test_images[i, c, :, :])
                if std > 0:
                    test_images[i, c, :, :] = (test_images[i, c, :, :] - mean) / std
    else:
        # Rescale pixel values from [0, 255] to [0, 1]
        train_images /= 255.0
        test_images /= 255.0

    ### END CODE HERE
    return train_images, test_images


class LeNet(nn.Module):
    '''
    Build the LeCun network according to the architecture in the homework part 4(c)

    You are free to use the listed APIs from torch.nn:
        torch.nn.Conv2d
        torch.nn.MaxPool2d
        torch.nn.Linear
        torch.nn.ReLU (or other activations)
        torch.nn.BatchNorm2d
        torch.nn.BatchNorm1d
        torch.nn.Dropout

    Refer to https://pytorch.org/docs/stable/nn.html
    for the instructions for those APIs
    '''
    def __init__(self, n_classes=None):
        super(LeNet, self).__init__()
        '''
        Define each layers of the model in __init__() function
        '''

        ### YOUR CODE HERE
        super(LeNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        # Feature map size calculation:
        # Input: 32x32x3
        # After conv1 (with padding=2): 32x32x6
        # After pool1: 16x16x6
        # After conv2 (no padding): 12x12x16
        # After pool2: 6x6x16

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)

        self.dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(84, n_classes)

        ### END CODE HERE
    
    def forward(self, x):
        '''
        Run forward pass of the model defined in the above __init__() function
        Args:
            x: Tensor of shape [None, 3, 32, 32]
            for input images.

        Returns:
            logits: Tensor of shape [None, n_classes].
        '''

        ### YOUR CODE HERE
        # First convolutional block
        x = self.conv1(x)
        #comment out these for experiment 2
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten feature maps
        x = x.reshape(-1, 16 * 6 * 6)

        # First fully connected block
        x = self.fc1(x)
        #x = self.bn3(x)
        x = self.relu(x)

        # Second fully connected block
        x = self.fc2(x)
        #x = self.bn4(x)
        x = self.relu(x)

        # Dropout
        #run1 include this, run 2 exclude this
        #x = self.dropout(x)

        #run2
        #None


        # Output layer
        x = self.fc3(x)

        ### END CODE HERE
        return x


class LeNet_Cifar10(nn.Module):
    def __init__(self, n_classes):

        super(LeNet_Cifar10, self).__init__()

        self.n_classes = n_classes
        self.model = LeNet(n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, max_epoch):

        num_samples = x_train.shape[0]
        num_batches = int(num_samples / batch_size)

        num_valid_samples = x_valid.shape[0]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train)
        x_valid = torch.from_numpy(x_valid).float()
        y_valid = torch.from_numpy(y_valid)

        print('---Run...')
        for epoch in range(1, max_epoch + 1):
            self.model.train()
            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()

                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[start:end]
                y_batch = curr_y_train[start:end]

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f}'.format(
                            epoch, loss))

            # To start validation at the end of each epoch.
            self.model.eval()
            correct = 0
            total = 0
            print('Doing validation...', end=' ')
            with torch.no_grad():
                for i in range(num_valid_batches):

                    start = batch_size * i
                    end = min(batch_size * (i + 1), x_valid.shape[0])
                    x_valid_batch = x_valid[start:end]
                    y_valid_batch = y_valid[start:end]

                    outputs = self.model(x_valid_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_valid_batch.shape[0]
                    correct += (predicted == y_valid_batch).sum().item()

            acc = correct / total
            print('Validation Acc {:.4f}'.format(acc))

    def test(self, X_test, y_test):
        self.model.eval()

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test)

        accs = 0
        for X, y in zip(X_test, y_test):

            outputs = self.model(X.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            accs += (predicted == y).sum().item()

        accuracy = float(accs) / len(y_test)
        
        return accuracy