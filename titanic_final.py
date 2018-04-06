# ML Data preprocessing template 
# importing libarary 


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing database 
dataset= pd.read_csv('train.csv')
X=dataset.drop(['Name','Ticket','Cabin','Embarked','Survived','Fare'],axis=1).values
Y=dataset.iloc[:,1].values

# Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, [3]])
X[:,[3]] = imputer.transform(X[:,[3]])


#Handling the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])


#Data spliting 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Implementing the random forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=12,criterion = 'entropy',random_state=0
                                    )
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Creating the test set 
Test= pd.read_csv('test.csv')
Test=Test.drop(['Name','Ticket','Cabin','Embarked','Fare'],axis=1).values

#Categotical data handling 
labelencoder_X2 = LabelEncoder()
Test[:, 2] = labelencoder_X2.fit_transform(Test[:, 2])

#Dealing with missing data 
imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer2 = imputer2.fit(Test[:, [3]])
Test[:,[3]] = imputer2.transform(Test[:,[3]])

#Final result computaion 
Prediction = classifier.predict(Test)

#make csv file 
submission = pd.DataFrame({ 'PassengerId': Test[:,0],
                            'Survived': Prediction })
submission.to_csv("submission.csv", index=False)

























