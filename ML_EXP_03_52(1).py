# Prajyot Pawar
# Roll no. 52
# LAB 03 :Ensemble Learning/ Bagging algorithm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv("D:\\Sem 7\\ML\\EXPTS\\3\\diabetes.csv")
df.head()

X = df.drop("Outcome", axis="columns")
y = df.Outcome

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, random_state=10)

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
bag_model.oob_score_

bag_model.score(X_test, y_test)

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, X, y, cv=5)
scores

print(scores.mean())
