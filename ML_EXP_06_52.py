# Prajyot Pawar
# Roll no. 52
# LAB06 : PCA/SVD/LDA
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

cancer_data = datasets.load_breast_cancer()
print(cancer_data.data[5])
print(cancer_data.data.shape)

print(cancer_data.target)

X_train, X_test, y_train, y_test = train_test_split(
    cancer_data.data, cancer_data.target, test_size=0.4, random_state=109)

cls = svm.SVC(kernel='linear')
cls.fit(X_train, y_train)
pred = cls.predict(X_test)

print("accuracy: ", metrics.accuracy_score(y_test, y_pred=pred))
print("precision: ", metrics.precision_score(y_test, y_pred=pred))
print("recall: ", metrics.recall_score(y_test, y_pred=pred))
print(metrics.classification_report(y_test, y_pred=pred))
