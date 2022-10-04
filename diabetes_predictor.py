import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv("H:/Datasets/Python Projects/Diabetes Prediction/diabetes.csv")

#df.head()

#df.describe()

#df['Outcome'].value_counts()

# 0 refers to non diabetic patients and 1 refers to diabetic patients

#sns.pairplot(df)

df[df['Outcome']==0].describe()

df[df['Outcome']==1].describe()

non_diabetic = df[df['Outcome']==0]
diabetic = df[df['Outcome']==1]

# Comparing the Glucose and Blood Pressure levels in Non-Diabetic vs Diabetic Patients

plt.figure(figsize=[10,6])
plt.scatter(non_diabetic.Glucose, non_diabetic.BloodPressure, marker='+')
plt.scatter(diabetic.Glucose, diabetic.BloodPressure, marker='x')

# Splitting the data

X = df.drop('Outcome', axis='columns')
y = df['Outcome']

# Scaling the data

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

standardized_data

X = standardized_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)

# Training the model

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

# Checking the accuracy on the training data

training_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(training_prediction, y_train)

print("Accuracy score of the training data : ", training_accuracy)

# Checking the accuracy on the test data

test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(test_prediction, y_test)

print("Accuracy score of the test data : ", test_accuracy)

# Making the predictive system

# Taking the input from the user
input_data = (1,81,72,18,40,26.6,0.283,24)

# We will need to change the data to numpy array
input_data_to_array = np.asarray(input_data)

# We will also need to reshape the input data as it will get different number of input parameters as compared to the training
reshaped_input_data = input_data_to_array.reshape(1,-1)

# As the model was trained on a scaled training data, we will need to scale the input data as well
standardized_input_data = scaler.transform(reshaped_input_data)

prediction = classifier.predict(standardized_input_data)

if prediction[0] == 0:
    print("The person in non-diabetic")
else:
    print("The person is diabetic")

# Saving the trained model

import pickle

filename = 'diabetes_model.sav'
model_dict = {'classifier': classifier, "scaler": scaler}
pickle.dump(model_dict, open(filename, 'wb'))

# Loading the model

loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))