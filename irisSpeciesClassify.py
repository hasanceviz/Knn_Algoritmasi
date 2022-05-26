import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Iris.csv')
del df["Id"]
df.head()

iris_csv = df.replace({'Iris-setosa': "1"})
iris_csv = df.replace({'Iris-versicolor': "2"})
iris_csv = df.replace({'Iris-virginica': "3"})

X = np.array(iris_csv.iloc[:, 0:4])
Y = np.array([[iris_csv.Species]])
Y = Y.reshape(150)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.32, random_state=0)

k=10
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("KNN için doğruluk puanı: ", accuracy_score(y_test, y_pred))
