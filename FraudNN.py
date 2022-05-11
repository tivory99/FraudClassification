import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#import csv into dataset
dataset = pd.read_csv('CREDITFRAUDDATASET.csv')

#create input data and classifier data
X= dataset.drop(columns=['step', 'nameOrig', 'type', 'nameDest', 'isFlaggedFraud', 'isFraud'])
y= dataset.drop(columns=['step', 'amount', 'nameOrig', 'type', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'nameDest', 'oldbalanceDest', 'isFlaggedFraud'])

#set shape to be equal
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#split data into testing and training datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#create model
from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=5))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#compile neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#run model
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

#evaluate model accuracy
eval_model=classifier.evaluate(X_train, y_train)
eval_model

#create confusion matrix
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#create visualization of neural network
from ann_visualizer.visualize import ann_viz
import os
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin/"
ann_viz(classifier, title="Fraud Detection Neural Network")