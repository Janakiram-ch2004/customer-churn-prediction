import numpy as np
import pandas as pd
import tensorflow as tf
df = pd.read_csv(r'C:\Users\rahul\OneDrive\Desktop\mini project\customer_prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df
df.shape
df.describe()
df.isna().sum()

df.duplicated().sum()
df.drop('customerID',axis=1,inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

num_cols = df.select_dtypes(include=['int64',"float64"])
cat_cols = df.select_dtypes(include='object')

num_cols

cat_cols

import matplotlib.pyplot as plt
import seaborn as sns


for col in num_cols:
  plt.figure(figsize=(10,5))
  sns.histplot(df[col], kde=True)
  plt.title(col)
  #plt.show() 

for col in num_cols:
  plt.figure(figsize=(10,5))
  sns.boxplot(df[col])
  plt.title(col)
  #plt.show()

for col in cat_cols:
  plt.figure(figsize=(10,5))
  sns.countplot(x=df[col])
  plt.title(col)
  #plt.show()

for col in cat_cols:
  plt.figure(figsize=(10,5))
  sns.countplot(x=df[col], hue=df['Churn'])
  plt.title(col)
  #plt.show()  

for col in num_cols:
  plt.figure(figsize=(10,5))
  sns.boxplot(x=df['Churn'], y=df[col])
  plt.title(col)
  #plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



x = df.drop('Churn',axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)



x_train_num = X_train.select_dtypes(include=['int64',"float64"])
x_train_cat = X_train.select_dtypes(include=["object"])

#x_train_cat

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x_train_num.columns),
        ('cat', OneHotEncoder(), x_train_cat.columns)
    ])

x_train_preprocessed = preprocessor.fit_transform(X_train)
x_test_preprocessed = preprocessor.transform(X_test)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train_preprocessed.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

y_train = y_train.map({'Yes': 1, 'No': 0})
y_test = y_test.map({'Yes': 1, 'No': 0})

model.fit(x_train_preprocessed, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping])


import joblib


joblib.dump(preprocessor, 'preprocessor.pkl')
print(" SUCCESS: preprocessor.pkl file is created!")


model.save('churn_dl_model.keras')
print(" SUCCESS: churn_dl_model.keras file is created!")










































