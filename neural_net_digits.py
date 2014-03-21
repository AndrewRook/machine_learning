import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

#Adapted from scikit-learn documentation.
def nudge_dataset(X, Y,dimen=(8,8)):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape(dimen), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def neural_net():
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    sidelength = int(np.sqrt(X.shape[1]))
    X,Y = nudge_dataset(X,digits.target,dimen=(sidelength,sidelength))
    #Scale the data to be between zero and 1 at all pixels:
    X = (X - np.min(X,axis=0))/(np.max(X,axis=0)+0.0001)

    #Split the data set into a training and testing set:
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    #The classifier
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)

    # Training Logistic regression
    #logistic_classifier = linear_model.LogisticRegression(C=100.0)
    #logistic_classifier.fit(X_train, Y_train)

    ###############################################################################
    # Evaluation
    print ""
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))
    
    #Predict a few individual cases:
    print classifier.predict(X_test[:5,:]),Y_test[:5]

    # print("Logistic regression using raw pixel features:\n%s\n" % (
    #     metrics.classification_report(
    #         Y_test,
    #         logistic_classifier.predict(X_test))))

    ###############################################################################


if __name__ == "__main__":
    neural_net()


