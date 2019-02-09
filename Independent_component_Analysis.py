# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:21:28 2018

@author: darvishm
"""

import pandas as pd

if __name__== '__main__':
 

    IMU_data = pd.read_csv("C:\Local\darvishm\one csv file mean 240 column small 4000 sample\A12IMUdata2.csv", 
                           encoding='utf-8')
    
    
    IMU_df = pd.DataFrame(IMU_data)
   
    
    IMU_df.head()
    IMU_df.info()
    IMU_df.describe()
    
    
   ############################################# IMU1 velocity and acceleration ###################################################
   def column_list(size):
        
        return ["Column" + str(i) for i in range(1,size + 1)]
 
   
   
   xv1 = IMU_data.iloc[:, 0:10]
    yv1 = IMU_data.iloc[:, 10:20]
    zv1 = IMU_data.iloc[:, 20:30]
    
    yv1.columns = column_list(10)
    zv1.columns = column_list(10)
    
    xa1 = IMU_data.iloc[:, 30:40]
    ya1 = IMU_data.iloc[:, 40:50]
    za1 = IMU_data.iloc[:, 50:60]
    
    
    xa1.columns = column_list(10)
    ya1.columns = column_list(10)
    za1.columns = column_list(10)
    
    ############################################# IMU2 velocity and acceleration ###################################################
    
    xv2 = IMU_data.iloc[:, 60:70]
    yv2 = IMU_data.iloc[:, 70:80]
    zv2 = IMU_data.iloc[:, 80:90]
    
    
    xv2.columns = column_list(10)
    yv2.columns = column_list(10)
    zv2.columns = column_list(10)
    
    
    xa2 = IMU_data.iloc[:, 90:100]
    ya2 = IMU_data.iloc[:, 100:110]
    za2 = IMU_data.iloc[:, 110:120]
    
    
    
    xa2.columns = column_list(10)
    ya2.columns = column_list(10)
    za2.columns = column_list(10)
    
    
    ################ IMU3 velocity and acceleration#################################
    xv3 = IMU_data.iloc[:, 120:130]
    yv3 = IMU_data.iloc[:, 130:140]
    zv3 = IMU_data.iloc[:, 140:150]
    
   
    xv3.columns = column_list(10)
    yv3.columns = column_list(10)
    zv3.columns = column_list(10)
    
    
    xa3 = IMU_data.iloc[:, 150:160]
    ya3 = IMU_data.iloc[:, 160:170]
    za3 = IMU_data.iloc[:, 170:180]
    
    xa3.columns = column_list(10)
    ya3.columns = column_list(10)
    za3.columns = column_list(10)
    
    
    ######################IMU4 acceleration and velocity #############################################
    
    xv4 = IMU_data.iloc[:, 180:190]
    yv4 = IMU_data.iloc[:, 190:200]
    zv4 = IMU_data.iloc[:, 200:210]
    
    xv4.columns = column_list(10)
    yv4.columns = column_list(10)
    zv4.columns = column_list(10)
    
    
    
    xa4 = IMU_data.iloc[:, 210:220]
    ya4 = IMU_data.iloc[:, 220:230]
    za4 = IMU_data.iloc[:, 230:240]
   
    xa4.columns = column_list(10)
    ya4.columns = column_list(10)
    za4.columns = column_list(10)
    
    
    ################################################ Classification ###############################################
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import cross_validation
    from sklearn.cross_validation import cross_val_score
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    
    df_label = IMU_data["Class"]
    
    df_data =  IMU_data.drop(["Class"], axis=1)
        
    
    x_train, x_test, y_train, y_test = train_test_split(df_data.values, df_label.values, 
                                                        test_size=0.2, random_state=0)
    
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    

        
        
        
######################################## ICA ############################################        
    

    from sklearn.decomposition import FastICA

    ica = FastICA(random_state=0)
    #fitted_model= ica.fit(x_train)
    
    x_train = ica.fit_transform(x_train)
    x_test = ica.transform(x_test)
    x_train.shape
    x_test.shape
    
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    accuracies = []
    for num_components in range(1, 241):
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
    
    accuracies = []
    for num_components in range(1, 241):
        clf = LogisticRegression()
        clf.C = 0.1
        clf.penalty = "l1"
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
    
    accuracies = []
    for num_components in range(1, 241):
        
        xg_cl = XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 10, alpha = 10, n_estimators = 100)
    
        xg_cl.fit(x_train[:, :num_components],y_train)
        y_pred = xg_cl.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
#        accuracy = cross_val_score(xg_cl,x_train,y_train, cv=10)
#        score=np.mean(accuracy)
#        accuracies.append(score)
        
    
    from sklearn.ensemble import RandomForestClassifier
    
    accuracies = []
    for num_components in range(1, 241):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
    
    from sklearn.svm import SVC
    
    accuracies = []
    for num_components in range(1, 241):
        clf = SVC(kernel = 'poly', C = 1e05)
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
        
        
    from sklearn.ensemble import ExtraTreesClassifier
    
    accuracies = []
    for num_components in range(1, 241):
        clf = ExtraTreesClassifier(n_estimators = 100)
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
        
        
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    
    accuracies = []
    for num_components in range(1, 241):
        clf = AdaBoostClassifier(n_estimators = 100)
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
        
    accuracies = []
    for num_components in range(1, 241):
        clf = GradientBoostingClassifier(n_estimators = 100)
        clf.fit(x_train[:, :num_components], y_train)
        y_pred = clf.predict(x_test[:, :num_components])
        accuracy = accuracy_score(y_test, y_pred)*100
        accuracies.append(accuracy)
        
    print(max(accuracies))
    print(accuracies.index(max(accuracies)))
    
        
        
        
        
    ####################### Plot ICA #######################################
    
    score = []
    for i in range(0, len(accuracies)):
        score.append(accuracies[i]*100)
    
    
    import matplotlib.pyplot as plt
    import os
    
    num_components = range(1, 200)
    
    plt.plot(num_components, score)
    
    plt.xlabel('Number of ICA components')
    plt.ylabel('Accuracy Percentage')
    plt.title('Classification accuracy')
    plt.grid(True)
    
    path = "P:\\Accuracy Plot\\"
    plt.savefig(os.path.join(path, 'Accuracy_xgb.pdf'))
#    plt.show()
        
    
    import matplotlib.pyplot as plt
    import os
    
    def plot_ICA_acuuracy(score, title, filename):
        
        num_components = range(1, 241)
    
        plt.plot(num_components, score)
        
        plt.xlabel('Number of ICA components')
        plt.ylabel('Accuracy Percentage')
        plt.title(title)
        plt.grid(True)
        
        path = "C:\\Local\\darvishm\\Accuracy Plot\\"
        plt.savefig(os.path.join(path, filename))
        
    
    
    plot_ICA_acuuracy(score, "XGBoost Classification accuracy", 'Accuracy_xgb.pdf')
        
    plot_ICA_acuuracy(accuracies, "LDA Classification accuracy", 'Accuracy_lda.pdf')
    
    plot_ICA_acuuracy(accuracies, "Logistic Regression Classification accuracy", 'Accuracy_log.pdf')
    
    plot_ICA_acuuracy(accuracies, "Random Forest Classification accuracy", 'Accuracy_rf.pdf')
    
    plot_ICA_acuuracy(accuracies, "SVM Classification accuracy", "Accuracy_svm.pdf")
    
    plot_ICA_acuuracy(accuracies, "Extra tree Classification accuracy", "Accuracy_extratree.pdf")
    
    plot_ICA_acuuracy(accuracies, "Adaboost Classification accuracy", "Accuracy_adaboost.pdf")
    
    plot_ICA_acuuracy(accuracies, "Gradient Boosting Classification accuracy", "Accuracy_gradboosttree.pdf")
    
