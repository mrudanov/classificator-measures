import pandas
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


# label, score1, score2, score3, score4, score5, score6, score7
DATAFILE = "classes.csv"


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.get_cmap('coolwarm')):
    plt.figure()
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=17)
    plt.yticks(tick_marks, classes, fontsize=17)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=18)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)


def create_confusion_matrix(true, predicted, class_number):
    func = np.vectorize(lambda x: 1 if x == class_number else 0)
    true_for_class = func(true)
    predicted_for_class = func(predicted)
    return confusion_matrix(true_for_class, predicted_for_class)


def transform_matrix_single_class(cm):
    transformed = [[0, 0], [0, 0]]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        transformed[i][j] = cm[1 - i, 1 - j]
    return np.array(transformed)


def split_confusion_matrix(cm):
    return cm[0][0], cm[1][1], cm[0][1], cm[1][0]


data_array = pandas.read_csv(DATAFILE)
y_predicted = np.array(data_array['predicted'])
y_true = np.array(data_array['label'])

# Prepare confusion matrices
cm_multiclass = confusion_matrix(y_true, y_predicted)
cm_class1 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 1))
cm_class2 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 2))
cm_class3 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 3))
cm_class4 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 4))
cm_class5 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 5))
cm_class6 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 6))
cm_class7 = transform_matrix_single_class(create_confusion_matrix(y_true, y_predicted, 7))

# Plot confusion matrices
plot_confusion_matrix(
    cm_multiclass,
    classes=['no', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7'],
    title='Multiclass confusion matrix')
plot_confusion_matrix(cm_class1, classes=['Positive', 'Negative'], title='Class 1')
plot_confusion_matrix(cm_class2, classes=['Positive', 'Negative'], title='Class 2')
plot_confusion_matrix(cm_class3, classes=['Positive', 'Negative'], title='Class 3')
plot_confusion_matrix(cm_class4, classes=['Positive', 'Negative'], title='Class 4')
plot_confusion_matrix(cm_class5, classes=['Positive', 'Negative'], title='Class 5')
plot_confusion_matrix(cm_class6, classes=['Positive', 'Negative'], title='Class 6')
plot_confusion_matrix(cm_class7, classes=['Positive', 'Negative'], title='Class 7')

# Perform calculations
K = 7
n = 29

tp1, tn1, fn1, fp1 = split_confusion_matrix(cm_class1)
tp2, tn2, fn2, fp2 = split_confusion_matrix(cm_class2)
tp3, tn3, fn3, fp3 = split_confusion_matrix(cm_class3)
tp4, tn4, fn4, fp4 = split_confusion_matrix(cm_class4)
tp5, tn5, fn5, fp5 = split_confusion_matrix(cm_class5)
tp6, tn6, fn6, fp6 = split_confusion_matrix(cm_class6)
tp7, tn7, fn7, fp7 = split_confusion_matrix(cm_class7)

err = (fp1 + fn1 + fp2 + fn2 + fp3 + fn3 + fp4 + fn4 + fp5 + fn5 + fp6 + fn6 + fp7 + fn7) / (K * n)
acc = 1 - err

sens_micro = (tp1 + tp2 + tp3 + tp4 + tp5 + tp6 + tp7) \
             / (tp1 + tp2 + tp3 + tp4 + tp5 + tp6 + tp7 + fn1 + fn2 + fn3 + fn4 + fn5 + fn6 + fn7)
sens_macro = ((tp1 / (tp1 + fn1))
              + (tp2 / (tp2 + fn2))
              + (tp3 / (tp3 + fn3))
              + (tp4 / (tp4 + fn4))
              + (tp5 / (tp5 + fn5))
              + (tp6 / (tp6 + fn6))
              + (tp7 / (tp7 + fn7))) / K

spec_micro = (tn1 + tn2 + tn3 + tn4 + tn5 + tn6 + tn7) \
             / (tn1 + tn2 + tn3 + tn4 + tn5 + tn6 + tn7 + fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7)
spec_macro = ((tn1 / (tn1 + fp1))
              + (tn2 / (tn2 + fp2))
              + (tn3 / (tn3 + fp3))
              + (tn4 / (tn4 + fp4))
              + (tn5 / (tn5 + fp5))
              + (tn6 / (tn6 + fp6))
              + (tn7 / (tn7 + fp7))) / K

fall_out_micro = 1 - spec_micro
fall_out_macro = 1 - spec_macro

prec_micro = (tp1 + tp2 + tp3 + tp4 + tp5 + tp6 + tp7) \
             / (tp1 + tp2 + tp3 + tp4 + tp5 + tp6 + tp7 + fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7)
prec_macro = ((tp1 / (tp1 + fp1))
              + (tp2 / (tp2 + fp2))
              + (tp3 / (tp3 + fp3))
              + (tp4 / (tp4 + fp4))
              + (tp5 / (tp5 + fp5))
              + (tp6 / (tp6 + fp6))
              + (tp7 / (tp7 + fp7))) / K

f1_score_micro = 2 * (prec_micro * sens_micro) / (prec_micro + sens_micro)

prec1 = (tp1 / (tp1 + fp1))
prec2 = (tp2 / (tp2 + fp2))
prec3 = (tp3 / (tp3 + fp3))
prec4 = (tp4 / (tp4 + fp4))
prec5 = (tp5 / (tp5 + fp5))
prec6 = (tp6 / (tp6 + fp6))
prec7 = (tp7 / (tp7 + fp7))

sens1 = (tp1 / (tp1 + fn1))
sens2 = (tp2 / (tp2 + fn2))
sens3 = (tp3 / (tp3 + fn3))
sens4 = (tp4 / (tp4 + fn4))
sens5 = (tp5 / (tp5 + fn5))
sens6 = (tp6 / (tp6 + fn6))
sens7 = (tp7 / (tp7 + fn7))

f1_score_macro = 2 * ((prec1 * sens1) / (prec1 + sens1)
                      + (prec2 * sens2) / (prec2 + sens2)
                      + (prec3 * sens3) / (prec3 + sens3)
                      + (prec4 * sens4) / (prec4 + sens4)
                      + (prec5 * sens5) / (prec5 + sens5)
                      + (prec6 * sens6) / (prec6 + sens6)
                      + (prec7 * sens7) / (prec7 + sens7)) / K

kappa = cohen_kappa_score(y_predicted, y_true)

# Print calculations
print("ERR:", err)
print("ACC:", acc)
print("SENS micro:", sens_micro)
print("SENS macro:", sens_macro)
print("SPEC micro:", spec_micro)
print("SPEC macro:", spec_macro)
print("PREC micro:", prec_micro)
print("PREC macro:", prec_macro)
print("Fall-out micro:", fall_out_micro)
print("Fall-out macro:", fall_out_macro)
print("F1-score micro:", f1_score_micro)
print("F1-score macro:", f1_score_macro)
print("Cohen's kappa:", kappa)

plt.show()
