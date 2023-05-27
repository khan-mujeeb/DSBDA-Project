import pandas as pd
data = pd.read_csv(r'F:\codes\dsbda\project\diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Accuracy:", lr.score(X_test, y_test))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

import pickle
from sklearn.svm import SVC
pickle.dump(lr, open('khan.pkl', 'wb'))
import numpy as np

# Create a new example dataset
example_data = np.array([[1, 185, 66, 29, 0, 26.6, 0.351, 31]])

# Preprocess the example data
example_data_processed = scaler.transform(example_data)

# Make a prediction on the example data
prediction = rf.predict(example_data_processed)

# Print the prediction
print(prediction)
