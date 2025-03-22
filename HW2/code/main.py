import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE

    # X should have shape [n_samples, 2] at this point

    plt.figure(figsize=(8, 6))
    #Error Found on Thursday Fixed on Sunday
    # Fix: Use correct column indices (0 and 1 since we're plotting 2D features)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', label='Class 2')
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig('train_features.png')
    plt.close()

    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE

    # Plot data points
    plt.figure(figsize=(8, 6))

    # Create a scatter plot of the data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='.', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b', marker='.', label='Class 2')

    # Create the decision boundary
    # We need to find points that satisfy W[0] + W[1]*x + W[2]*y = 0
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x_points = np.linspace(x_min, x_max, 100)

    # From W[0] + W[1]*x + W[2]*y = 0, we get:
    # y = -(W[0] + W[1]*x)/W[2]
    y_points = -(W[0] + W[1] * x_points) / W[2]

    # Plot the decision boundary
    plt.plot(x_points, y_points, 'k-', label='Decision Boundary')

    # Set labels and title
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()

    # Adjust the plot limits to show all data points
    plt.xlim(x_min, x_max)
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    plt.ylim(y_min, y_max)

    plt.savefig('train_result_sigmoid.png')
    plt.close()
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(10, 8))

    # Plot data points for each class
    colors = ['g', 'r', 'b']
    labels = ['Digit 0', 'Digit 1', 'Digit 2']
    # Plot data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='.', label='Class 1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='.', label='Class 2')

    # Create decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Add bias term
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]

    # Get predictions
    Z = np.argmax(np.dot(grid, W), axis=1)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, Z, colors='k', levels=[0.5])

    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set labels to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    ### YOUR CODE HERE

    # Add this right before visualize_features call
    train_y[train_y == 2] = -1  # Convert class 2 to -1
    valid_y[valid_y == 2] = -1  # Convert validation labels too

    ### END YOUR CODE
    data_shape = train_y.shape[0] 

    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   #### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    # Test different batch sizes and learning rates
    learning_rates = [0.1, 0.5, 1.0]
    batch_sizes = [1, 10, 50, 100]
    best_score = 0
    best_params = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            model = logistic_regression(learning_rate=lr, max_iter=100)
            model.fit_miniBGD(train_X, train_y, batch_size)
            score = model.score(valid_X, valid_y)

            if score > best_score:
                best_score = score
                best_params = (lr, batch_size)
                best_model = model

    print(f"Best parameters: learning_rate={best_params[0]}, batch_size={best_params[1]}")
    print(f"Best validation accuracy: {best_score}")

    ### END YOUR CODE

    ### YOUR CODE HERE

    # Visualize your 'best' model after training.
    visualize_result(train_X[:, 1:3], train_y, best_model.get_params())

    ### END YOUR CODE

    # Use the 'best' model above to do testing.
    # Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    # Test the model
    test_X, test_y = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(test_X)
    test_y, test_idx = prepare_y(test_y)
    test_X = test_X[test_idx]
    test_y = test_y[test_idx]
    test_y[test_y == 2] = -1

    test_accuracy = best_model.score(test_X, test_y)
    print(f"Test accuracy: {test_accuracy}")
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    # Multi-class classification
    # Multi-class classification
    print("\nTraining multi-class classifier...")
    learning_rates = [0.1, 0.5, 1.0]
    batch_sizes = [10, 50, 100]
    best_multi_score = 0
    best_multi_params = None

    # Use all data for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    for lr in learning_rates:
        for batch_size in batch_sizes:
            multi_model = logistic_regression_multiclass(learning_rate=lr, max_iter=1000, k=3)
            multi_model.fit_miniBGD(train_X, train_y, batch_size)
            score = multi_model.score(valid_X, valid_y)

            if score > best_multi_score:
                best_multi_score = score
                best_multi_params = (lr, batch_size)
                best_multi_model = multi_model

    print(f"Best multi-class parameters: learning_rate={best_multi_params[0]}, batch_size={best_multi_params[1]}")
    print(f"Best multi-class validation accuracy: {best_multi_score}")

    # Visualize multi-class results with all three digits
    visualize_result_multi(train_X[:, 1:3], train_y, best_multi_model.get_params())
    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE#
    # Comparison section for sigmoid vs softmax
    print("\nComparing Sigmoid and Softmax classifiers...")

    # Prepare data once
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]

    # Create separate copies for sigmoid and softmax
    sigmoid_train_y = train_y.copy()
    softmax_train_y = train_y.copy()

    # Convert labels appropriately
    sigmoid_train_y[sigmoid_train_y == 2] = -1  # Convert to -1, 1
    softmax_train_y[softmax_train_y == 2] = 0  # Convert to 0, 1

    # Train both classifiers
    sigmoid_classifier = logistic_regression(learning_rate=0.1, max_iter=1000)
    sigmoid_classifier.fit_miniBGD(train_X, sigmoid_train_y, batch_size=50)

    softmax_classifier = logistic_regression_multiclass(learning_rate=0.1, max_iter=1000, k=2)
    softmax_classifier.fit_miniBGD(train_X, softmax_train_y, batch_size=50)

    # Create comparison plot
    plt.figure(figsize=(15, 6))

    # Sigmoid subplot
    plt.subplot(1, 2, 1)
    X_plot = train_X[:, 1:3]  # Only use symmetry and intensity features
    visualize_result(X_plot, sigmoid_train_y, sigmoid_classifier.get_params())
    plt.title('Sigmoid Decision Boundary')

    # Softmax subplot
    plt.subplot(1, 2, 2)
    visualize_result_multi(X_plot, softmax_train_y, softmax_classifier.get_params())
    plt.title('Softmax Decision Boundary')

    plt.tight_layout()
    plt.savefig('comparison_boundaries.png')
    plt.close()
    ################Compare and report the observations/prediction accuracy

    # ------------End------------


if __name__ == '__main__':
    main()
    
    
