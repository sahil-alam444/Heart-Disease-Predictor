import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection & Processing"""

heart_data = pd.read_csv('heart_data.csv')

# first 5 datas
heart_data.head()

# last 5 datas
heart_data.tail()

heart_data.shape

heart_data.info()

heart_data.isnull().sum()

# statistical measure about data

heart_data.describe()

# checking the distribution of Target variable

heart_data['target'].value_counts()

"""1 --> Defective Heart
0 --> Healthy Heart

---

Splitting Features & Targets
"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""Splitting Data into **Training Data** & **Test Data**"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""**Model Training : Logistic Regression**"""

model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

"""**Model Evaluation - Accuracy Score**"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training Data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test Data : ', test_data_accuracy)

"""Building a **Predictive System**"""

input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)

# change the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy arrays we are predicting for only on instance

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person have a Heart Disease')

input_data = (55, 1, 0, 160, 289, 0, 0, 145, 1, 0.8, 1, 1, 3)

# change the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy arrays we are predicting for only on instance

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person doesn't have any Heart Disease")
else:
    print('The Person have some Heart Disease')
