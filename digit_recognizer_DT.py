import pandas as pd
import numpy as np
train = pd.read_csv('/Users/Liang/Downloads/train.csv')
X = train.drop('label', axis=1).values
y = train['label'].values

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

clf = DecisionTreeClassifier(criterion='entropy', random_state=100, min_samples_leaf=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))