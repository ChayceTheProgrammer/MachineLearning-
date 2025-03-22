import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Class for Digit Classification.

    This class implements a feedforward neural network with the following architecture:
    - Input Layer: Accepts input features (e.g., extracted from images).
    - Hidden Layers: Two fully connected layers with ReLU activation functions.
    - Output Layer: A fully connected layer that outputs logits for three classes (0, 1, 2).

    The network is designed to classify input data based on features extracted from the raw data.
    The architecture includes dropout layers for regularization to help prevent overfitting during training.

    Attributes:
        fc1 (nn.Linear): First fully connected layer mapping input features to hidden units.
        fc2 (nn.Linear): Second fully connected layer mapping hidden units to more hidden units.
        fc3 (nn.Linear): Output layer mapping hidden units to output classes.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, input_size=3, hidden_size=3, output_size=3):
        super(MLP, self).__init__()
        ### YOUR CODE HERE
        """
        Initializes the MLP model with specified input, hidden, and output sizes.

        Args:
            input_size (int): Number of input features (default is 3).
            hidden_size (int): Number of neurons in the hidden layers (default is 128).
            output_size (int): Number of output classes (default is 3).
        """
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization

        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        """
        Defines the forward pass of the MLP.

        The input tensor is passed through the network layers, applying ReLU activation
        and dropout between the hidden layers. The final output is the raw logits from
        the output layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size).
        """

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        ### END YOUR CODE
        return x
