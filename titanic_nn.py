
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv('titanic.csv') 
print(data.head())
print(data.columns)

#  Data Exploration (EDA)

print("\nDataset info:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

print("\nSurvived value counts:")
print(data['Survived'].value_counts())


# Data Preprocessing

# Separate features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Drop irrelevant columns
X.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# Encode categorical variables
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])  # male=1, female=0

le_embarked = LabelEncoder()
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])  # 0,1,2

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Define Neural Network Architecture

input_dim = X_train.shape[1]  # number of features
model = Sequential()

# Input layer + first hidden layer
model.add(Dense(units=16, activation='relu', input_dim=input_dim))

# Second hidden layer
model.add(Dense(units=8, activation='relu'))

# Output layer (binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Step 7: Compile the Model
# ==============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Step 8: Train the Model
# ==============================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Make Predictions 

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # convert probabilities to 0 or 1
print("\nFirst 10 predictions:\n", y_pred[:10])

# Save model
model.save('titanic_model.h5')
print("\nModel saved as 'titanic_model.h5'")
