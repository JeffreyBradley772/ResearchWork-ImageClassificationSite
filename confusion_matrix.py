#code from https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
import matplotlib.pyplot as plt 
import numpy as np
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ):
    """
    This function prints and plots the confusion matrix.
    """

    import itertools

    new_cm = [[0 for x in range(len(cm))] for y in range(len(cm))]
    for i in range(len(cm)):
        total = float(sum(cm[i]))
        for j in range(len(cm)):
            new_cm[i][j] = cm[i][j] / total

    new_cm = np.array(new_cm)
    print(new_cm)
    print(cm)

    plt.imshow(new_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = new_cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(new_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if new_cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()