# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:23:21 2019

@author: bahar
"""

################################################ Classification ###############################################
import pandas as pd
import scipy.io as spio
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

feature_df = pd.read_csv('P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\New feature dataset based on acceleration whole data\\new_feature_dataset_whole_original.csv')
IMU_class = pd.read_csv('P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\Class_dataset_whole.csv')

feature_df = feature_df.drop("Unnamed: 0", axis=1)
IMU_class = IMU_class.drop("Unnamed: 0", axis=1)



##### splitting data into train and test

x_train = feature_df.iloc[0:600349 , :]
x_test = feature_df.iloc[600349: , :]

y_train = IMU_class.iloc[0:600349 , :]
y_test = IMU_class.iloc[600349: , :]


label = np.ravel(IMU_class)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)



######################################## logistic regression ########################################################################################


start = time.time()
clf_l = LogisticRegression()
C_range = 10.0 ** np.arange(-6, 2) 
for C in C_range:
    for penalty in ["l1", "l2"]:
        clf_l.C = C
        clf_l.penalty = penalty
        clf_l.fit(x_train, y_train)
        y_pred = clf_l.predict(x_test)
        accuracy = 100.0 * accuracy_score(y_test, y_pred)
        print ("Score/C/Penalty", accuracy, C, penalty)

end = time.time()
print(end - start)
        
        
start = time.time()
clf = LogisticRegression()
clf.C = 1.0
clf.penalty = "l2"
clf.fit(x_train, y_train)

with open('logloss.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_pred = clf.predict(x_test)
ac=accuracy_score(y_test, y_pred)
print(ac)
end = time.time()
print(end - start)


start = time.time()
clf = LogisticRegression()
accuracy = cross_val_score(clf,feature_df.values, label, cv=10, n_jobs = 1)
print(accuracy)
score=np.mean(accuracy)*100 
print(score)
end = time.time()
print(end - start)


#####################################################Logistic regression#####################################################




########################################################### KNN ####################################################
start = time.time()

clf = KNeighborsClassifier(metric = "euclidean")
clf.fit(x_train.values, y_train)
y_pred=clf.predict(x_test.values)
k_range = [3,5,7,11,13]
for k in k_range:
    clf.kneighbors = k
    accuracy = accuracy_score(y_test, y_pred)
    print ("Score/kneighnours", accuracy, k)
end = time.time()
print(end - start)





start = time.time()

clf = KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")
clf.fit(x_train, y_train)

#  save it to a file
with open('KNN.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_pred=clf.predict(x_test)
ac=accuracy_score(y_test, y_pred)
print(ac*100)
accuracy = cross_val_score(clf, df_data.values, label,cv=10)
print(accuracy)
score=np.mean(accuracy)*100
print(score)

end = time.time()
print(end - start)


####################################################################### KNN ################################################



###############################################################RandomForest classifier################################################
from sklearn.ensemble import RandomForestClassifier

start = time.time()
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)

with open('RF.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc*100)
accuracy = cross_val_score(clf, df_data.values, label,cv=10)
print(accuracy)
score=np.mean(accuracy)*100
print(score)
end = time.time()
print(end - start)

####################################################################### LDA ##########################################################################


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

start = time.time()
clf = LDA()
clf.fit(x_train, y_train)

with open('LDA.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_pred=clf.predict(x_test)
ac_lda = accuracy_score(y_test, y_pred)
print(ac_lda*100)
end = time.time()
print(end - start)



start = time.time()
clf = LDA()
accuracy = cross_val_score(clf, feature_df.values, label, cv=10)
print(accuracy)
score=np.mean(accuracy)*100
print(score)
end = time.time()
print(end - start)



###############################################################xgboost ##################################################################
start = time.time()
xg_cl = XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 10, alpha = 10, n_estimators = 100)

xg_cl.fit(x_train.values,y_train)

with open('XGB.pkl', 'wb') as f:
    pickle.dump(xg_cl, f)

y_pred = xg_cl.predict(x_test.values)

ac = accuracy_score(y_test, y_pred)
print(ac*100)
end = time.time()
print(end - start)


start = time.time()
accuracy = cross_val_score(xg_cl, feature_df.values, IMU_class, cv=10)
print(accuracy)
score=np.mean(accuracy)*100
print(score)
end = time.time()
print(end - start)

 

 ##############################################################################################################   
##########################################################svm ###########################################################




from sklearn.svm import SVC


start = time.time()
# C is the penalty term (default = 1)
clf = SVC(kernel = 'poly', degree=3, C = 1)
clf.fit(x_train, y_train)

with open('svm.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_pred = clf.predict(x_test)
ac_svm = accuracy_score(y_test, y_pred)
print(ac_svm*100)

accuracy = cross_val_score(clf, df_data, label, cv = 10)
print(accuracy)
score=np.mean(accuracy)*100
print(score)
end = time.time()
print(end - start)


#######################Parameter tunning svm ##########################################

from sklearn.model_selection import GridSearchCV

start = time.time()
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
      'gamma': [0.0001, 0.001, 0.01, 0.1],
      'kernel':['linear','poly','rbf'] }

grid_clf = GridSearchCV(SVC(), param_grid= params, n_jobs= 2)

grid_clf = grid_clf.fit(x_train, y_train)

with open('grid_clf.pkl', 'wb') as f:
    pickle.dump(grid_clf, f)

print(grid_clf.best_estimators)

end = time.time()

print(end-start)


# comparison of tree classifiers
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

classifiers = [(RandomForestClassifier(), "Random Forest"),
                (ExtraTreesClassifier(), "Extra-Trees"),
                (AdaBoostClassifier(), "AdaBoost"),
                (GradientBoostingClassifier(), "GB-Trees")]


for clf, name in classifiers:
    clf.n_estimators = 100
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    cv_scores = cross_val_score(clf , x_train , y_train  , n_jobs=1, cv=10)
    print( name, cv_scores.mean()*100)
     

