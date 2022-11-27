# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235554/189818813-d8bd8e14-affa-4f38-bb80-09c2518c9bc6.png)

## DESIGN STEPS

### STEP 1:
Loading the dataset.

### STEP 2:
Checking the null values and converting the string datatype into integer or float datatype using label encoder.

### STEP 3:
Split the dataset into training and testing.

### STEP 4:
Create MinMaxScaler objects,fit the model and transform the data.

### STEP 5:
Build the Neural Network Model and compile the model.

### STEP 6:
Train the model with the training data.

### STEP 7:
Plot the training loss and validation loss.

### STEP 8:
Predicting the model through classification report,confusion matrix.

### STEP 9:
Predict the new sample data.

## PROGRAM
```python
# Developed By: Harshini M
# Register Number: 212220230022
import pandas as pd
import numpy as np

df=pd.read_csv("customers.csv")


df.head()

df=df.dropna(axis=0)

df["Segmentation"].unique()

import tensorflow as tf

x=df.drop(["Segmentation","Var_1","ID"],axis=1)
y=df[["Segmentation"]].values


df["Gender"].unique(),df["Ever_Married"].unique(),df["Graduated"].unique(),df["Profession"].unique(),df["Spending_Score"].unique()

from sklearn.preprocessing import OneHotEncoder

lr=OneHotEncoder()

lr.fit(y)

lr.categories_

y=lr.transform(y).toarray()

y

from sklearn.preprocessing import OrdinalEncoder

lst=[['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor','Homemaker', 'Entertainment', 'Marketing', 'Executive'],['Low', 'High', 'Average']]

enc = OrdinalEncoder(categories=lst)

x[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = enc.fit_transform(x[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])

x=x.drop(["Age","Work_Experience","Family_Size"],axis=1)

x=x.values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.10,
                                               random_state=50)

model=tf.keras.Sequential([tf.keras.layers.Input(shape=(5,)),
                           tf.keras.layers.Dense(128,activation="relu"),
                           tf.keras.layers.Dense(64,activation="relu"),
                           tf.keras.layers.Dense(32,activation="relu"),
                           tf.keras.layers.Dense(4,activation="softmax")])

model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test),batch_size=32)

pd.DataFrame(model.history.history).plot()

y_preds=tf.argmax(model.predict(X_test),axis=1)

import sklearn

sklearn.metrics.classification_report(tf.argmax(y_test,axis=1),y_preds)

sklearn.metrics.confusion_matrix(tf.argmax(y_test,axis=1),y_preds)

tf.argmax(model.predict([[0., 0., 0., 6., 0.]]),axis=1)
```

## Dataset Information

<img width="560" alt="image" src="https://user-images.githubusercontent.com/75235554/189571938-78567d6c-e0bf-4498-b0c7-da5f3978705e.png">

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235554/189819105-ab13f99a-4d88-49c4-8d50-b3a94c95ef2e.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235554/189819148-ed19ed56-d619-4772-912f-1002e389ed3d.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235554/189819226-a5876083-6fae-4b12-a044-2f1ae1f043ba.png)

### New Sample Data Prediction

<img width="410" alt="image" src="https://user-images.githubusercontent.com/75235554/189819292-09e3720d-7609-48ef-8336-088b3a68a33d.png">

## RESULT
Thus a Neural Network Classification Model is created and executed successfully.
