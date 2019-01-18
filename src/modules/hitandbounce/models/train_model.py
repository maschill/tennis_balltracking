# Train model for hit and bounce detection
import pandas as pd
import numpy as np
from joblib import dump
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import criticalPoints

#############################################
# Generate features
#############################################
# Number of points used for regression
nump = 3
# Because of smoothing edges disappear. Jump can be used to leave points out
jump = 1
# load smoothed ball positions
with open('../../../../data/processed/pickle_smoothed_ball_position.pkl', 'rb') as file:
    smballpos = pickle.load(file)

bouncehitfeatures = criticalPoints.genFeatures(smpos=smballpos, nump=nump, jump=jump)

# Transform in pandas dataframe prepare it and add more features
bouncehitfeatures = pd.DataFrame(bouncehitfeatures)
bouncehitfeatures[[1, 2, 3, 4, 5, 6, 7]] = bouncehitfeatures[[1, 2, 3, 4, 5, 6, 7]].apply(pd.to_numeric)
bouncehitfeatures = bouncehitfeatures.set_index(0)
bouncehitfeatures.columns = ['angle', 'sc1', 'sc2', 'r1_x', 'r1_y', 'r2_x', 'r2_y']
bouncehitfeatures['sum_sc'] = bouncehitfeatures['sc1'] + bouncehitfeatures['sc2']

#############################################
# Load annotated data
#############################################
hitservedata = []
with open('../../../../data/annotations/BounceHit.txt', 'r') as f:
    for row in f:
        num, label = row.strip('\n').split(' ')[0:2]
        hitservedata += [num, label]
hitservedata = np.array(hitservedata).reshape(-1, 2)
hitservedata = pd.DataFrame(hitservedata)
hitservedata = hitservedata.set_index(0)
hitservedata.columns = ['label']

#############################################
# Prepare training and test set
#############################################
hitservedata = hitservedata.join(bouncehitfeatures, how='inner')
train, test = train_test_split(hitservedata)

#############################################
# Train model
#############################################
features = ['angle','sc1','sc2','r1_x','r1_y','r2_x','r2_y','sum_sc']

clf = RandomForestClassifier(n_estimators=30,
                             max_depth=20,
                             max_features=8,
                             random_state=0,
                             class_weight='balanced')

clf.fit(train[features], train['label'])

print('Importance of features ', features, '\n', clf.feature_importances_)
print(train['label'].value_counts())
print('    ', 'Bounce', 'Hit', 'Nothing')
print(confusion_matrix(test['label'], clf.predict(test[features])))
print(clf.score(test[features], test['label']))

dump(clf, 'hitandbouncedetectionmodel.joblib')
