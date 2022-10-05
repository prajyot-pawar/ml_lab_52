# Prajyot Pawar
# Roll no. 52
# LAB 02 : Logistic Regression
import matplotlib.pyplot as plt
import numpy as np

n = float(input("Enter no of values:"))
x1 = [float(k) for k in input("Enter %d values for x1:\n" % n).split(",")]
x2 = [float(k) for k in input("Enter %d values for x2:\n" % n).split(",")]
y = [float(k) for k in input("Enter %d values for y:\n" % n).split(",")]
print(x1, x2)
print(y)
# Prajyot Pawar
# Roll no. 52
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
b0 = b1 = b2 = 0
s = 0.3
p = []
pc = []
for i in range(int(n)):
    p.append(1/(1+np.exp(-(b0+b1*x1[i]+b2*x2[i]))))
    b0 = b0 + s * (y[i]-p[i])*p[i]*(1-p[i])*1
    b1 = b1 + s * (y[i]-p[i])*p[i]*(1-p[i])*x1[i]
    b2 = b2 + s * (y[i]-p[i])*p[i]*(1-p[i])*x2[i]
    if (p[i] > 0.5):
        pc.append(1)
    else:
        pc.append(0)
print("X1              X2         Actual Class         Prediction          Predicted Class")
for i in range(int(n)):
    print('%f      %f              %d                %f                     %d' % (
        x1[i], x2[i], int(y[i]), p[i], pc[i]))
