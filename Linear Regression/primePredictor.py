import pickle as p
import numpy as np
ob = p.load(open('./num.p', 'rb'))

X = np.array(ob['data'])
X = X.reshape(-1, 1)

Y = ob['class']

from sklearn.model_selection import train_test_split

Xtr, xte, ytr, yte = train_test_split(X, Y, random_state= 101, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

import os
clf = RandomForestClassifier()
if not 'classifier.p' in os.listdir():
    clf.fit(Xtr, ytr)
    p.dump(clf, open('classifier.p', 'wb'))
else:
    clf = p.load(open('classifier.p', 'rb'))


pred = clf.predict(xte)

from sklearn.metrics import accuracy_score

acc = accuracy_score(yte, pred)

print(acc*100, '%')
