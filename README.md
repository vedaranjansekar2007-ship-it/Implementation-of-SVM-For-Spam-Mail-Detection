# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load the dataset.
2. Separate input messages and output labels.
3. Split data into training and testing sets.
4. Convert text into numerical form and train the SVM model.
5. Predict results and evaluate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VEDARANJAN S
RegisterNumber:  212225220119
*/

import pandas as pd
data=pd.read_csv(r"C:\Users\acer\Downloads\spam.csv",encoding='latin -1') 
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

Data <br>

<img width="1211" height="682" alt="image" src="https://github.com/user-attachments/assets/bdaf7a1f-248b-4508-84b8-d8322ca638c1" />

Accuracy <br>

<img width="377" height="62" alt="image" src="https://github.com/user-attachments/assets/6fd48a7e-4379-435b-86d4-5e7f9cb9e19e" />

Confusion Matrix <br>

<img width="316" height="95" alt="image" src="https://github.com/user-attachments/assets/1b5684ab-be8c-41a8-ac08-0e274f15f4ed" />

Report <br>

<img width="930" height="288" alt="Screenshot 2026-05-20 113121" src="https://github.com/user-attachments/assets/40662949-7eab-41ac-a6d1-204821001dfd" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
