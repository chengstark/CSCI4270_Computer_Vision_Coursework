import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import svm, datasets
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


labels = os.listdir('hw6_data/train')[1:]
log = ''
sampling = False


# helper function for plotting confusion matrix, this is a standard function from sklearn documentation
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    global log
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
        log += "Normalized confusion matrix\n"
    else:
        print('Confusion matrix, without normalization')
        log += 'Confusion matrix, without normalization\n'
    print(cm)
    log += str(cm)
    log += '\n'

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


# train the SVM
@ignore_warnings(category=ConvergenceWarning)
def train_svm():
    global log
    # load training set
    train_ = np.loadtxt('train_vectors.txt')
    # loop through all classes
    for i in range(len(labels)):
        print('Currently training for {} class'.format(labels[i]))
        current_train = train_.copy()
        # translate the data set in to one vs rest by parsing labels to 1 and 0
        this_class = current_train[current_train[:, -1] == i]
        other_classes = current_train[current_train[:, -1] != i]
        this_class[:, -1] = 1
        other_classes[:, -1] = 0
        # combine the parsed training data
        new_train = np.concatenate((this_class, other_classes))
        X = new_train[:, :-1]
        Y = new_train[:, -1]
        scores = []
        # test different c
        for c in range(1, 11):
            clf = LinearSVC(C=c, class_weight='balanced')
            # cross validate
            score = cross_val_score(clf, X, Y, cv=10)
            print('\tC {} <-> score {}'.format(c, score.mean()))
            scores.append(score.mean())
        # get best c based on score
        best_c_idx = scores.index(max(scores))
        best_c = range(1, 11)[best_c_idx]
        print('Best C is {}, best score is {}'.format(best_c, max(scores)))
        # plot confidence values vs validation set
        plt.clf()
        plt.plot(range(1, 11), scores)
        plt.xlabel('C')
        plt.ylabel('cross_val_score')
        plt.title('C selection for class {}'.format(labels[i]))
        plt.savefig('SVM_c_tune/c_tune_{}.jpg'.format(labels[i]))
        # use the best parameter to train a svm and save it
        best_clf = LinearSVC(C=best_c)
        best_clf.fit(X, Y)
        pickle.dump(best_clf, open('SVM_clfs/{}_clf.pkl'.format(labels[i]), 'wb'))

        print('Finished class {}'.format(labels[i]))


# helper function to get all image path
def get_test_img_paths():
    all_paths = []
    for label_idx in range(len(labels)):
        folder_path = 'hw6_data/test/{}/'.format(labels[label_idx])
        image_names = os.listdir(folder_path)
        for im_name in image_names:
            path = os.path.join(folder_path, im_name)
            all_paths.append(path)
    return all_paths


# calculate confidence of each test image
def calc_confidence(row, clf):
    return (np.dot(clf.coef_, row) + clf.intercept_).flatten()


# test the SVM
def test_svm():
    global log
    # get predictions
    test_ = np.loadtxt('test_vectors.txt')
    test_X = test_[:, :-1]
    test_Y = test_[:, -1]
    test_X = normalize(test_X, axis=1)
    # initialize confidence recorder
    # initialize values to -infinity to make sure does not interfere with later real values
    all_confidence = np.zeros_like(test_Y).reshape(test_Y.shape[0], 1)
    all_confidence -= np.inf
    for i in range(len(labels)):
        print('Currently testing {} class'.format(labels[i]))
        # load SVMs
        with open('SVM_clfs/{}_clf.pkl'.format(labels[i]), 'rb') as f:
            clf = pickle.load(f)
        # calculate confidence values of each test image
        confidence = np.apply_along_axis(calc_confidence, 1, test_X, clf)
        confidence.reshape(confidence.shape[0], 1)
        # combine all confidence values
        all_confidence = np.dstack((all_confidence, confidence))

    # get the most confidence one from all candidates
    test_pred = np.argmax(all_confidence, axis=2).flatten()
    test_pred -= 1
    # calculate test accuracy
    test_acc = (test_Y == test_pred).sum() / float(test_Y.size)
    print('Test accuracy: {}'.format(test_acc))
    log += 'Test accuracy: {}\n'.format(test_acc)
    # plot confusion matrix
    plot_confusion_matrix(test_Y, test_pred, classes=labels, normalize=True,
                          title='Test confusion matrix')
    plt.savefig('SVM_test_matrix.jpg')

    # if we want to sample the results
    if sampling:
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
        # save good predictions and bad predictions to images
        all_test_paths = get_test_img_paths()
        for i in range(len(test_Y)):
            if test_pred[i] != test_Y[i]:
                bad_pred[test_pred[i]].append(all_test_paths[i])
            else:
                good_pred[test_pred[i]].append(all_test_paths[i])
        # save sampled images
        os.mkdir('SVM_result_samples')
        for label in labels:
            os.mkdir('SVM_result_samples/{}'.format(label))
            os.mkdir('SVM_result_samples/{}/good'.format(label))
            os.mkdir('SVM_result_samples/{}/bad'.format(label))
        # loop through good classification samples and bad classification samples
        for i in range(len(labels)):
            good_counter = 0
            bad_counter = 0
            for a in good_pred[i]:
                img = cv2.imread(a)
                tokens = a.split('/')
                cv2.imwrite('SVM_result_samples/{}/good/{}_{}.jpg'.format(labels[i], good_counter, tokens[-1][:-5]), img)
                good_counter += 1
                if good_counter == 100:
                    break
            for b in bad_pred[i]:
                img = cv2.imread(b)
                tokens = b.split('/')
                cv2.imwrite('SVM_result_samples/{}/bad/{}_{}.jpg'.format(labels[i], bad_counter, tokens[-1][:-5]), img)
                bad_counter += 1
                if bad_counter == 100:
                    break


if __name__ == '__main__':
    # train SVM and test SVM
    train_svm()
    test_svm()
