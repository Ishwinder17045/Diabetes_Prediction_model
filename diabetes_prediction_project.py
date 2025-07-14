import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('training_data.csv')

diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

import numpy as np

# Ask the user to enter each value
print("Please enter the following health details:")

pregnancies = int(input("Number of Pregnancies: "))
glucose = int(input("Glucose Level: "))
blood_pressure = int(input("Blood Pressure value: "))
skin_thickness = int(input("Skin Thickness value: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI value: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function value: "))
age = int(input("Age: "))

# Create a tuple with all values
input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)

# Convert to numpy array and reshape for the model
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

# Make prediction
prediction = classifier.predict(std_data)

# Output result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')