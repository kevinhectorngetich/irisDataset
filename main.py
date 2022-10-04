import sys
import numpy, matplotlib, pandas, sklearn, scipy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# we are using panda to load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)  # total number of row columns

# head
print(dataset.head(20))

print(dataset.describe())  # print mean std min max 50%

print(dataset.groupby('class').size())

# --== Uni variate plot =--
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()

# Histogram
# dataset.hist()
# plt.show()

# Multi Variate plot
#    scatter matrix
# scatter_matrix(dataset)
# plt.show()

# create some algorithm
# Validation data set -- used to train your model
# Split you data into 2 sets
# e.g. first 80% will use it to train our data 20% for validation

array = dataset.values
# print('Array values / contents', str(array))
X = array[:, 0:4]  # starting from 0 to 4
# print('X variate values', str(X))
Y = array[:, 4]
# print('Y variate values: ', str(Y))
validation_size = 0.20
seed = 6  # starting value in generating random number
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# so the seed is helping in keeping the same randomness in the training and testing data set

# Test Harness
# Use 10-fold cross validation to estimate the accuracy
# it splits the data sets into 10 parts train on the 9 parts and test on the one part.

seed = 6
scoring = 'accuracy'

# Building the Model
# we don't know which algorithm would be usefull for this, so we use the following 6

# Spot check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s:  %f  (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# LDA was the most accurate