# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Load the csv file and then use the preprocessing steps to clean the data

### STEP 2:
Split the data to training and testing

### STEP 3:
Train the data and then predict using Tensorflow

## PROGRAM
```python
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

Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here


### New Sample Data Prediction

Include your sample input and output here

## RESULT
Thus a Neural Network Classification Model is created and executed successfully.
