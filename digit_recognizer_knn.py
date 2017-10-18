import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
train = pd.read_csv('/Users/Liang/Downloads/train.csv')
X = train.drop('label', axis=1).values
X = scale(X)
y = train['label'].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


knn = KNeighborsClassifier(n_neighbors=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))