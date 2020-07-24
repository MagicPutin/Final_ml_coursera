import pandas as pd
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

start_time = datetime.datetime.now()
"""# data preparing

X_test = pd.read_csv('Data/features_test.csv', index_col='match_id')

features = pd.read_csv('Data/features.csv', index_col='match_id')
X = features.loc[:, 'start_time':'dire_first_ward_time']
y = features['radiant_win']

# 1st stage
gap_date = X.count() - X.shape[0]
print(gap_date.loc[gap_date < 0] * (-1))  # columns with gap
print('\npercentage of gap date: ' + str(round(sum(gap_date * (-1))/(X.shape[0] * X.shape[1]), 2)))

X = X.fillna(0)  # fill gaps with nouns
# X = X.fillna(features.mean()) # fill gaps with mean

kf = KFold(n_splits=5, shuffle=True)

n_forest = 100 # for 3rd question
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



print('\nTime elapsed:', datetime.datetime.now() - start_time)