import pandas as pd
import numpy as np
import datetime

# data preparing
start_time = datetime.datetime.now()

features = pd.read_csv('Data/features.csv', index_col='match_id').loc[:, 'start_time':'dire_first_ward_time']
gap_date = features.count() - features.shape[0]
print(gap_date.loc[gap_date < 0] * (-1))  # columns with gap
print('\npercentage of gap date: ' + str(round(sum(gap_date * (-1))/(features.shape[0] * features.shape[1]), 2)))

# features = features.fillna(0)  # fill gaps with nouns

# features = features[features.isnull()] = np.nanmean([features.shift(-48), features.shift(48)])
# print(features.count())

print('\nTime elapsed:', datetime.datetime.now() - start_time)

# quality metric
# pred = clf.predict_proba(X_test)[:, 1]
