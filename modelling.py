from timeit import default_timer as timer
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# new
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# classifiers = {
#     'MLPC': MLPClassifier(solver='adam', activation='tanh',
#                           early_stopping=False, max_iter=100,
#                           alpha=0.01, hidden_layer_sizes=(128, 128, 128)),
#     'MLPR': MLPRegressor(solver='adam', activation='tanh',
#                          early_stopping=False, max_iter=100,
#                          alpha=0.01, hidden_layer_sizes=(128, 128, 128)),
#     'RF': RandomForestClassifier(n_estimators=100),
#     'SVM': SVC(decision_function_shape='ovo')
# }


#################################################################

# def train_model(X, y,
#                 label_ids: list, method='MLPC', test_train = None):
    
#         # process the input data so it is normalised
#         scaler = StandardScaler()
#         scaler.fit(X)
#         X_std = scaler.transform(X)

#         # train using the selected method
#         start = timer()
#         model = classifiers[method].fit(X_std, y)
#         end = timer()
#         time_to_train = end - start

#         return model, (scaler, timer)


# def validate_model(model, scaler, X_test, y_test):

#     #scale given validation set
#     X_test_std = scaler.transform(X_test)

#     # test the performance of the model on the test set
#     y_pred = model.predict(X_test_std)
#     # convert numpy string array to numpy int array of class indices (MLPC has
#     # slightly different interface)
#     y_pred = np.fromiter(map(lambda x: np.where(model.classes_ == x)[
#         0] if isinstance(x, str) else x, y_pred), dtype=np.int)
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred)

#     return accuracy, confusion_mat

# # K-fold CV
# def kfold_CV(X,y,label_ids,method,k=5):
    
#     kf = KFold(n_splits=k, shuffle=True, random_state=0)
#     confs = []
#     for train_index, test_index in kf.split(X):
        
#         # train
#         X_train = X[train_index]
#         y_train = y[train_index]

#         # test
#         X_test = X[test_index]
#         y_test = y[test_index]

#         # model
#         model, stats = train_model(X_train, y_train, label_ids=list(label_ids.keys()), method=method)
#         acc, cm = validate_model(model, stats[0], X_test, y_test)

#         confs.append(cm) # record matrix
        
#     return confs, acc

# confusion matrix tools

def normalise_confusion_matrix(cm):
    return (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

def plot_confusion_matrix(cm, label_ids: dict,
                          percented = False,
                          title = None,
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    
    if cm.shape[0] != len(label_ids):
        raise Exception('Number of labels does not match confusion matrix!')
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    
    # get classes for labels
    ids = list(label_ids.keys())
    class_labels = [label_ids[i]+', '+str(i) for i in ids]
    
    # We want to show all ticks...
    ax.set(xticks=ids, yticks=ids,
           # ... and label them with the respective list entries
           xticklabels=ids, yticklabels=class_labels,
           title=title,
           ylabel='TRUE',
           xlabel='PREDICTED')

    # Rotate the tick labels and set their alignment.
    # ax.xaxis.tick_top()
    # plt.setp(ax.get_xticklabels(), rotation=-20, ha="right", va = "bottom", rotation_mode="anchor")

    thresh = cm.max() / 2.
    
    # Loop over data dimensions and create text annotations.
    if percented:
        cm = cm.astype(int)
        for (j,i),value in np.ndenumerate(cm):
            ax.text(i,j,str(value),ha='center',va='center',color='white' if value > thresh else ('white' if value == 0 else 'black'))
    else:
        for (j,i),value in np.ndenumerate(cm):
            ax.text(i,j,value,ha='center',va='center',color='white' if value > thresh else ('white' if value == 0 else 'black'))
    fig.tight_layout()
    return ax