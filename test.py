#### importing library


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# importing and preparing data

data = pd.read_csv('iris.csv')
rows, cols = data.shape
dataset = np.array(data)
all_vals = dataset[:, 0:4]
all_labels_string = dataset[:, 4:5]
all_labels = np.unique(all_labels_string, return_inverse=True)[1]


# Creating function


# Basic functions required for the test

def second_element(elem):
    return elem[1]


# getting neighbor indices

def neighbors(test_row, train_rows, number_neighbors):
    distances = list()
    train_row_index = -1
    for train_row in train_rows:
        train_row_index += 1
        distance = 0
        for i in range(len(train_row)):
            distance += (train_row[i] - test_row[i]) ** 2

        distances.append([train_row_index, distance])

    distances.sort(key=second_element)

    n_indices = [index[0] for index in distances[0:number_neighbors]]
    return n_indices


######  a.


# making predictions for test data

def knnclassify(test_data, training_data, training_labels, K):
    predictions = list()
    for test_data_row in test_data:
        neighbor_indices = neighbors(test_data_row, training_data, K)
        neighbor_labels = [training_labels[i] for i in neighbor_indices[:]]
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        predictions.append(prediction)

    return predictions


###### b.


# Defining matrix number_iteration times number_K

accuracy_matrix = np.zeros([8, 100])

# Accuracy calculation

for m in range(100):

    (training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)

    for n in range(1, 9):
        pred_lables = knnclassify(test_data, training_data, training_labels, n)
        accuracy = sum(test_labels == pred_lables) / len(test_labels)
        accuracy_matrix[(n - 1), m] = accuracy

# Average Accuracy calculation

avg_accuracy = np.array([np.mean(i) for i in accuracy_matrix])
avg_accuracy_rounded = np.round(avg_accuracy, 3)
sd_accuracy = np.array([np.std(i) for i in accuracy_matrix])
sd_accuracy_rounded = np.round(sd_accuracy, 3)
print(avg_accuracy_rounded)
print(sd_accuracy_rounded)

# Plotting

import matplotlib.pyplot as plt

X = list(range(1, 9))
plt.rcParams['figure.figsize'] = (19, 10)
plt.errorbar(X, avg_accuracy_rounded, sd_accuracy_rounded, ecolor='black')
plt.title('KNN Accuracy with STD', fontsize=20)
plt.xlabel('number of K', fontsize=15)
plt.ylabel('Avg Accuracy', fontsize=15)

for x, y in zip(X, avg_accuracy_rounded):
    plt.annotate(sd_accuracy_rounded[x - 1], (x, y), textcoords="offset points", xytext=(-10, 10), ha='right')

plt.show()

###### c.


(training_data_1, test_data_1, training_labels_1, test_labels_1) = train_test_split(all_vals, all_labels, test_size=0.3)

#### K-NN Classification with only Petal Length and Sepal Width


train_data_PL_SW = training_data_1[:, (2, 1)]
test_data_PL_SW = test_data_1[:, (2, 1)]
pred_lables_1 = knnclassify(test_data_PL_SW, train_data_PL_SW, training_labels_1, 1)
accuracy_1 = sum(test_labels_1 == pred_lables_1) / len(test_labels_1)

#### K-NN Classification with only Sepal Length and Sepal Width


train_data_SL_SW = training_data_1[:, (0, 1)]
test_data_SL_SW = test_data_1[:, (0, 1)]
pred_lables_2 = knnclassify(test_data_SL_SW, train_data_SL_SW, training_labels_1, 1)
accuracy_2 = sum(test_labels_1 == pred_lables_2) / len(test_labels_1)

# avg_accuracy_1 = np.mean(all_accuracy_1)
# avg_accuracy_2 = np.mean(all_accuracy_2)


print('accuracy in case of KNN Classification with only PL and SW is ', accuracy_1)
print('accuracy in case of KNN Classification with only SL and SW is ', accuracy_2)
