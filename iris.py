from sklearn import datasets
import pandas as pd
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

def calc_mis_classification_err(y, predictions):
    mis_class_count = 0
    mis_class_instances = []
    correct_predictions = {0: 0, 1: 0, 2: 0}

    for i in range(len(predictions)):
        if y[i] != predictions[i]:
            mis_class_instances.append((y[i], predictions[i]))
            mis_class_count += 1
        else:
            correct_predictions[y[i]] += 1
    
    return (mis_class_instances, mis_class_count, correct_predictions)

def calc_predictions(W, X):

    predictions = []

    for i in range(len(X)):
        result = np.dot(np.transpose(W), np.transpose(X[i]).reshape(5,1))
        max_el = result[0]
        max_idx = 0

        for j in range(1, len(result)):
            if result[j] > max_el:
                max_el = result[j]
                max_idx = j
        
        predictions.append(max_idx)
    
    return predictions

def calc_y_class_matrix(y):

    y_class_matrix = np.zeros((len(y), 3))

    for i in range(len(y)):
        if y[i] == 0:
            y_class_matrix[i] = [1,0,0]
        elif y[i] == 1:
            y_class_matrix[i] = [0,1,0]
        elif y[i] == 2:
            y_class_matrix[i] = [0,0,1]
        
    return y_class_matrix

def create_confusion_matrix(mis_class_instances, correct_predictions):

    confusion_matrix = np.zeros((3,3))

    for expected, predicted in mis_class_instances:
        confusion_matrix[int(predicted)][int(expected)] += 1

    confusion_matrix[0][0] = correct_predictions[0]
    confusion_matrix[1][1] = correct_predictions[1]
    confusion_matrix[2][2] = correct_predictions[2]

    return confusion_matrix

def split_train_test(X, y, ratio):

    num_per_class = int(ratio * 150) // 3 
    train_vector_size = int(ratio * 150)
    test_vector_size = 150 - int(ratio * 150)
    
    train_X = np.vstack((X[0:num_per_class,:], X[50:50+num_per_class,:], X[100:100+num_per_class,:]))
    train_y = np.vstack((y[0:num_per_class], y[50:50+num_per_class], y[100:100+num_per_class])).reshape(train_vector_size,)
    test_X = np.vstack((X[num_per_class:50,:], X[50+num_per_class:100,:], X[100+num_per_class:150,:]))
    test_y = np.vstack((y[num_per_class:50], y[50+num_per_class:100], y[100+num_per_class:150])).reshape(test_vector_size,)

    return train_X, test_X, train_y, test_y

# 1. Download the Iris data set
 
iris_dataset = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[iris_dataset['data'], iris_dataset['target']], columns = iris_dataset['feature_names'] + ['iris type'])
X = (iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]).as_matrix()
y = iris_df['iris type'].as_matrix()

# 2. Add column of 1's to X for bias in weight matrix

X = np.c_[X, np.ones((150, 1))]

# 3. Separate X and y into training and testing sets

training_ratios = (.12, .30, .50)
ratio = training_ratios[2] ### <---- Choose from the three options provided in training_ratios
train_X, test_X, train_y, test_y = split_train_test(X, y, ratio) 

# 4. Calculate W matrix

y_class_matrix = calc_y_class_matrix(train_y)

cross_val = 0.001
corr_mat = np.zeros((5,5))
x_y_mat = np.zeros((5,3))

for i in range(len(train_X)):
    corr_mat += np.outer(train_X[i], train_X[i])    
    x_y_mat += np.dot(np.transpose(train_X[i]).reshape(5,1), y_class_matrix[i].reshape(1,3))

W = np.matmul(inv(corr_mat + (cross_val * np.identity(5))), x_y_mat)

# 5. Predict classes for test set

train_predictions = calc_predictions(W, train_X)
test_predictions = calc_predictions(W, test_X)

# 6. Calculate the training & testing mis-classification errors and the correct predictions per each class

train_mis_class_instances, train_mis_class_count, train_correct_predictions = calc_mis_classification_err(train_y, train_predictions)
test_mis_class_instances, test_mis_class_count, test_correct_predictions = calc_mis_classification_err(test_y, test_predictions)

train_correct_predictions_count = 0
for key, item in train_correct_predictions.items():
    train_correct_predictions_count += item

test_correct_predictions_count = 0
for key, item in test_correct_predictions.items():
    test_correct_predictions_count += item

# 7. Output training/testing info and accuracy score & Make the confusion matrix 
print("------------------- CLASSIFIER INFO ----------------------------")
print(f"Number of Training instances: {len(train_y)}")
print(f"Number of Training mis-classification errors: {train_mis_class_count}" )
print(f"Training Accuracy score: { (train_correct_predictions_count / len(train_predictions)) * 100 } %")
print("----------------------------------------------------------------")
print(f"Number of Testing instances: {len(test_y)}")
print(f"Number of Testing mis-classification errors: {test_mis_class_count}" )
print(f"Testing Accuracy score: { (test_correct_predictions_count / len(test_predictions)) * 100 } %")

train_confusion_matrix = create_confusion_matrix(train_mis_class_instances, train_correct_predictions)
test_confusion_matrix = create_confusion_matrix(test_mis_class_instances, test_correct_predictions)

# 8. Plot the confusion matrix
row_labels = col_labels = ["setosa (0)", "versicolor (1)", "virginica (2)"]
plt.table(cellText=train_confusion_matrix, loc='center', cellLoc='center', colLabels= col_labels, rowLabels = row_labels)
ax = plt.gca()
ax.axis('off')
plt.title(f'Training Confusion Matrix ({ratio * 100}% Training / {100 - (ratio * 100)}% Testing)')
plt.show()

ax = plt.gca()
ax.axis('off')
plt.table(cellText=test_confusion_matrix, loc='center', cellLoc='center', colLabels= col_labels, rowLabels = row_labels)
plt.title(f'Testing Confusion Matrix ({ratio * 100}% Training / {100 - (ratio * 100)}% Testing)')
plt.show()
