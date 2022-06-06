import pandas as pd
import numpy as nm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Using pandas to read the dataset file in order to visualize it
dataset = pd.read_csv("C:/Users/ASUS/Desktop/FEUP/3ano/2Semestre/IART/PROJ-IART2/src/data.csv")

# Creating the correlation matrix in order to eliminate any variables with a correlation higher than 0.95 since they
# will have low impact on the improving of the algorithm
dataset_corr = dataset.corr().abs()
upper = dataset_corr.where(nm.triu(nm.ones(dataset_corr.shape), k=1).astype(bool))
# No collumns have a correlation higher than 0.95 so we won't remove any of them
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

#final_data = dataset.drop(columns=[""])

inputs = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Dividing the dataset into the train and test datasets
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.25, random_state=0)


ovsp = SMOTE(random_state=1)
unsp = RandomUnderSampler(random_state=1)

ovsp_inputs, ovsp_labels = ovsp.fitresample(inputs, labels)
print(Counter(ovsp_labels))

unsp_inputs, unsp_labels = unsp.fitresample(inputs, labels)
print(Counter(unsp_labels))



st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)


# Decision Trees Algorithm
#
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(x_train, y_train)
#
# y_pred = classifier.predict(x_test)
#
# cm = confusion_matrix(y_test, y_pred)
#
# # Calculating the accurracy of the Decision Tree algorithm
#
# print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
#
# # It is calculated by considering the total TP, total FP and total FN of the confusion matrix. It does not consider each class individually, It calculates the metrics globally
#
# print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
#
# # It calculates metrics for each class individually and then takes unweighted mean of the measures.
#
# print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
#
# #  Unlike Macro F1, it takes a weighted mean of the measures. The weights for each class are the total number of samples of that class.
#
# print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))


# # KKN
#
# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# # Fitting the classifier to the training data
# classifier.fit(x_train, y_train)
#
# # Predicting the test set result
# y_pred = classifier.predict(x_test)
#
# cm = confusion_matrix(y_test, y_pred)
#
# print(cm)
# # # Calculating the accuracy of the KKN Algorithm
# #
# print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
# #
#
# # It is calculated by considering the total TP, total FP and total FN of the confusion matrix. It does not consider each class (Graduate, Dropout, Enroled) individually, It calculates the metrics globally
#
# print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
#
# # It calculates metrics for each class individually and then takes unweighted mean of the measures.
#
# print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
#
# #  Unlike Macro F1, it takes a weighted mean of the measures. The weights for each class are the total number of samples of that class.
#
# print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
#
#
#

# # Support Vector Machines
#
# classifier = SVC(kernel='linear')
# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)
# print("Support Vector Machines accuracy: {:}".format(accuracy_score(y_test, y_pred)))
#
# cm = confusion_matrix(y_test, y_pred)

# MLP - Multilayer Perceptron

classifier = MLPClassifier(random_state=1, early_stopping=False)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("MLP - Multilayer Perceptron accuracy: {:}".format(accuracy_score(y_test, y_pred)))




# print(y_test)
# print(y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# cm_df = pd.DataFrame(cm,
#                      index = ['DROPOUT','GRADUATE','ENROLLED'],
#                      columns = ['DROPOUT','GRADUATE','ENROLLED'])

# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()
# Plotting of the graphs

# for col in dataset.columns[1:-1]:
#     plt.suptitle(col)
#     plt.xlabel(col)
#     plt.ylabel("Dropouts")
#     plt.scatter(dataset[col], dataset[dataset.columns[-1]])
#     plt.show()

# Decision trees


