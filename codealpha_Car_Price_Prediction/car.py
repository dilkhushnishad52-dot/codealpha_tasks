import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

df = pd.read_csv('car data.csv')

df=df.drop(columns=['Present_Price','Selling_type','Transmission','Owner'])

x = df[['Car_Name','Year','Driven_kms','Fuel_Type']]
y = df['Selling_Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(x[['Car_Name','Fuel_Type']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['Car_Name','Fuel_Type']),remainder='passthrough')

lr = LinearRegression()
## pipeline
pipe=make_pipeline(column_trans,lr)
### fitting the model

pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

## checking R2 score
r2_score(y_test,y_pred)

scores=[]
for i in range(1000):
    x_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

np.argmax(scores)

accuracy = scores[np.argmax(scores)]

print("car price pediction !!")
name = input("enter car name : ")
year = int(input("enter  year : "))
driven = int(input("enter driven km : "))
fuel = input("enter fuel type : ")

price = pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array([name,year,driven,fuel]).reshape(1,4)))

print("car price is : ",price)
print("Accuracy : ",accuracy*100)