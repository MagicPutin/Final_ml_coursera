import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

start_time = datetime.datetime.now()
# data preparing

features = pd.read_csv('Data/features.csv', index_col='match_id')
X = features.loc[:, 'start_time':'dire_first_ward_time']
y = features['radiant_win']

X = X.fillna(0)  # fill gaps with nouns
# X = X.fillna(features.mean()) # fill gaps with mean
"""
# 1st stage
gap_date = X.count() - X.shape[0]
print(gap_date.loc[gap_date < 0] * (-1))  # columns with gap
print('\npercentage of gap date: ' + str(round(sum(gap_date * (-1))/(X.shape[0] * X.shape[1]), 2)))
"""


kf = KFold(n_splits=5, shuffle=True)
"""
n_forest = 30 # for 3rd question
score = 0
clf = GradientBoostingClassifier(n_estimators=n_forest)
for train_index, test_index in kf.split(X): # splitting data via KFold and calculate metric
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    score += roc_auc_score(y_test, pred)
print(str(n_forest) + ' : ' + str(round(score/5, 2)))
"""
print('\nTime elapsed:', datetime.datetime.now() - start_time)

# 2nd stage
start_time = datetime.datetime.now()

N = 112
# bag of words
bag = np.zeros((features.shape[0], N))

for i, match_id in enumerate(features.index):
    for p in range(5):
        bag[i, features.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        bag[i, features.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

# dropping categorise features
X = X.drop(['lobby_type'], axis=1)
X = X.drop(columns=['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])

# maybe there's smth more smart, but i'm so lazy

# data scaling
scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X))
X = scaler.fit_transform(X)

# appending bag of word to df
"""X_pick = pd.DataFrame(X_pick.tolist())
X = pd.concat([X, X_pick], axis=1, sort=False)"""

X = np.hstack((X, bag))

clf = LogisticRegression(penalty='l2')

C = 0.1  # best parameter
score = 0
for train_index, test_index in kf.split(X):  # splitting data via KFold and calculate metric
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    score += roc_auc_score(y_test, pred)
print(str(C) + ' : ' + str(score / 5) + '\n')

# final stage (prediction on test data) and give csv file
test = pd.read_csv('Data/features_test.csv', index_col='match_id')
test = test.fillna(0)
bag = np.zeros((test.shape[0], N))
for i, match_id in enumerate(test.index):
    for p in range(5):
        bag[i, test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        bag[i, test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
test = test.drop(['lobby_type'], axis=1)
test = test.drop(columns=['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
test = scaler.fit_transform(test)
test = np.hstack((test, bag))
pred = clf.predict_proba(test)[:, 1]
pred = pd.Series(pred, name='radiant_win')
match_id = pd.Series(pd.read_csv('Data/features_test.csv')['match_id'], name='match_id')
match_id = pd.concat([match_id, pred], axis=1)
match_id.to_csv('Result/result.csv', index=False)

print('\nTime elapsed:', datetime.datetime.now() - start_time)