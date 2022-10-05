# Prajyot Pawar
# Roll no. 52
# LAB04 : MULTIVARIATE REGRESSION
from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('D:\\Sem 7\\ML\\EXPTS\\4\\homeprices.csv')

md = df.bedrooms.median()
df.bedrooms = df.bedrooms.fillna(md)

reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df.price)

print("coef:", reg.coef_)
print("intercept:", reg.intercept_)

print("predict (area=3000, bedrooms=3, age=40):", reg.predict([[3000, 3, 40]]))

print("predict (area=2500, bedrooms=4, age=5):", reg.predict([[2500, 4, 5]]))
