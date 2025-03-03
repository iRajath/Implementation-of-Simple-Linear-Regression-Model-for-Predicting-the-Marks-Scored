# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary libraries.
2. Initialize variables to assign dataset values.
3. Import the linear regression module from sklearn.
4. Specify the points to be represented on the graph.
5. Predict the regression of marks using the graphical representation.
6. Compare the graphs, resulting in the linear regression for the given data.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S Rajath
RegisterNumber:  212224240127
*/


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/0bc8e984-804f-43b4-a059-f7bd7640bf03)

![image](https://github.com/user-attachments/assets/a710cc0c-4a2b-426c-a1e1-cc88773e3bfd)

![image](https://github.com/user-attachments/assets/debde3e4-5ce4-4db8-8f09-f994cb57ce83)

![image](https://github.com/user-attachments/assets/db07b134-0e66-4921-8676-5a2dcd9e365c)

![image](https://github.com/user-attachments/assets/bd77bb82-97e0-4e4e-bf77-5ea4d0abe299)

![image](https://github.com/user-attachments/assets/a6817adb-a2d4-49f6-9435-95a0e85930b7)

![image](https://github.com/user-attachments/assets/0ec0f604-7022-4b31-854d-7c5318647b06)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
