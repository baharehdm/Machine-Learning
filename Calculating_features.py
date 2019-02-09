# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:21:28 2018

@author: darvishm
"""

import pandas as pd
import scipy.io as spio
import numpy as np
import pickle

if __name__== '__main__':
 
    mat = spio.loadmat('C:\\Local\\darvishm\\NEW DATASET\\Combined_mean_240col_whole.mat')
    X_train = mat["Train_X_mean"]
    Y_train = mat["Train_Y_mean"]
    
    
    
    mtype = X_train.dtype
    print(mtype)
    
    print(X_train.shape)
    print(Y_train.shape)
    
    y = np.array(Y_train) 
    print(y)
    
    x = np.array(X_train, dtype='f')
    
    print(x.shape)
    print(y.shape)
   
    def column_list(size):
        
        return ["Column" + str(i) for i in range(1,size + 1)]
    
    
    IMU_data = pd.DataFrame.from_records(x, columns = column_list(240))
    
    IMU_class = pd.DataFrame.from_records(y)

#    
    
   ############################################# IMU1 velocity and acceleration ###################################################
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
    
    
    
     #######################################angular velocities###########################################
    xv1_3 = xv1.sub(xv3, fill_value=0)
    
    xv1_4 = xv1.sub(xv4, fill_value=0)
    
    xv2_3 = xv2.sub(xv3, fill_value=0)
    
    xv2_4 = xv2.sub(xv4, fill_value=0)    
    
    
    yv1_3 = yv1.sub(yv3, fill_value=0)
    
    yv1_4 = yv1.sub(yv4, fill_value=0)
    
    yv2_3 = yv2.sub(yv3, fill_value=0)
    
    yv2_4 = yv2.sub(yv4, fill_value=0)
    
    
    zv1_3 = zv1.sub(zv3, fill_value=0)
    
    zv1_4 = zv1.sub(zv4, fill_value=0)
    
    zv2_3 = zv2.sub(zv3, fill_value=0)
    
    zv2_4 = zv2.sub(zv4, fill_value=0)
    
    ###################################linear accelations####################################################
    
    xa1_3 = xa1.sub(xa3, fill_value=0)
    
    xa1_4 = xa1.sub(xa4, fill_value=0)
    
    xa2_3 = xa2.sub(xa3, fill_value=0)
    
    xa2_4 = xa2.sub(xa4, fill_value=0)
    
    
    ya1_3 = ya1.sub(ya3, fill_value=0)
    
    ya1_4 = ya1.sub(ya4, fill_value=0)
    
    ya2_3 = ya2.sub(ya3, fill_value=0)
    
    ya2_4 = ya2.sub(ya4, fill_value=0)
    
    
    za1_3 = za1.sub(za3, fill_value=0)
    
    za1_4 = za1.sub(za4, fill_value=0)
    
    za2_3 = za2.sub(za3, fill_value=0)
    
    za2_4 = za2.sub(za4, fill_value=0)
    
     ####################################################### Difference vector feature ###################################################################################################
    
    
    
    # IMU1-3
    m1_10_1_3_x = xv1_3["Column1"].sub(xv1_3["Column10"], fill_value=0)
    m1_10_1_3_y = yv1_3["Column1"].sub(yv1_3["Column10"], fill_value=0)
    m1_10_1_3_z = zv1_3["Column1"].sub(zv1_3["Column10"], fill_value=0)
    
    m1_10_1_3 = pd.concat([m1_10_1_3_x, m1_10_1_3_y, m1_10_1_3_z], axis=1)
    m1_10_1_3 = m1_10_1_3.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    
    m1_5_1_3_x = xv1_3["Column1"].sub(xv1_3["Column5"], fill_value=0)
    m1_5_1_3_y = yv1_3["Column1"].sub(yv1_3["Column5"], fill_value=0)
    m1_5_1_3_z = zv1_3["Column1"].sub(zv1_3["Column5"], fill_value=0)
    
    m1_5_1_3 = pd.concat([m1_5_1_3_x, m1_5_1_3_y, m1_5_1_3_z], axis=1)
    m1_5_1_3 = m1_5_1_3.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    #IMU2-4
    m1_10_2_4_x = xv2_4["Column1"].sub(xv2_4["Column10"], fill_value=0)
    m1_10_2_4_y = yv2_4["Column1"].sub(yv2_4["Column10"], fill_value=0)
    m1_10_2_4_z = zv2_4["Column1"].sub(zv2_4["Column10"], fill_value=0)
    
    m1_10_2_4 = pd.concat([m1_10_2_4_x, m1_10_2_4_y, m1_10_2_4_z], axis=1)
    m1_10_2_4 = m1_10_2_4.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    
    m1_5_2_4_x = xv2_4["Column1"].sub(xv2_4["Column5"], fill_value=0)
    m1_5_2_4_y = yv2_4["Column1"].sub(yv2_4["Column5"], fill_value=0)
    m1_5_2_4_z = zv2_4["Column1"].sub(zv2_4["Column5"], fill_value=0)
    
    m1_5_2_4 = pd.concat([m1_5_2_4_x, m1_5_2_4_y, m1_5_2_4_z], axis=1)
    m1_5_2_4 = m1_5_2_4.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    
    
    
    
    feature_df = pd.DataFrame()
    
    feature_df["V1_10_1_3_x"] = m1_10_1_3["X"]
    feature_df["V1_10_1_3_y"] = m1_10_1_3["Y"]
    feature_df["V1_10_1_3_z"] = m1_10_1_3["Z"]
    
    feature_df["V1_5_1_3_x"] = m1_5_1_3["X"]
    feature_df["V1_5_1_3_y"] = m1_5_1_3["Y"]
    feature_df["V1_5_1_3_z"] = m1_5_1_3["Z"]
    
    
    
    feature_df["V1_10_2_4_x"] = m1_10_2_4["X"]
    feature_df["V1_10_2_4_y"] = m1_10_2_4["Y"]
    feature_df["V1_10_2_4_z"] = m1_10_2_4["Z"]
    
    feature_df["V1_5_2_4_x"] = m1_5_2_4["X"]
    feature_df["V1_5_2_4_y"] = m1_5_2_4["Y"]
    feature_df["V1_5_2_4_z"] = m1_5_2_4["Z"]
    
    
    
    ################################## Acceleration vector feature ##########################################################################
    mat_acc = spio.loadmat('C:\\Local\\darvishm\\NEW DATASET\\Combined_Acceleration_100len.mat')
    acc_IMU1 = mat_acc["acc_IMU1"]
    acc_IMU2 = mat_acc["acc_IMU2"]
    acc_IMU3 = mat_acc["acc_IMU3"]
    acc_IMU4 = mat_acc["acc_IMU4"]
    
    
    print(acc_IMU1.shape)
    print(acc_IMU2.shape)
    print(acc_IMU3.shape)
    print(acc_IMU4.shape)
    
    acc_IMU1 = np.array(acc_IMU1, dtype='f') 
    acc_IMU2 = np.array(acc_IMU2, dtype='f') 
    acc_IMU3 = np.array(acc_IMU3, dtype='f') 
    acc_IMU4 = np.array(acc_IMU4, dtype='f') 
    
    
    print(acc_IMU1.shape)
    print(acc_IMU2.shape)
    print(acc_IMU3.shape)
    print(acc_IMU4.shape)
    
    acc_IMU1 = pd.DataFrame.from_records(acc_IMU1)
    acc_IMU2 = pd.DataFrame.from_records(acc_IMU2)
    acc_IMU3 = pd.DataFrame.from_records(acc_IMU3)
    acc_IMU4 = pd.DataFrame.from_records(acc_IMU4)
    
    
    ####IMU1
    xacc1 = acc_IMU1.iloc[:, 0:100]
    yacc1 = acc_IMU1.iloc[:, 100:200]
    zacc1 = acc_IMU1.iloc[:, 200:300]
    
    #sum of columns
    acc1x_sum_IMU1 = xacc1.sum(axis=1)
    xacc1_50 = xacc1.iloc[:, 0:50]
    acc2x_sum_IMU1 = xacc1_50.sum(axis=1)
    
    #sum of columns
    acc1y_sum_IMU1 = yacc1.sum(axis=1)
    yacc1_50 = yacc1.iloc[:, 0:50]
    acc2y_sum_IMU1 = yacc1_50.sum(axis=1)
    
    #sum of columns
    acc1z_sum_IMU1 = zacc1.sum(axis=1)
    zacc1_50 = zacc1.iloc[:, 0:50]
    acc2z_sum_IMU1 = zacc1_50.sum(axis=1)
    
    ###### Features
    acc1_sum_IMU1 = pd.concat([acc1x_sum_IMU1, acc1y_sum_IMU1, acc1z_sum_IMU1], axis=1)
    acc1_sum_IMU1 = acc1_sum_IMU1.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    acc2_sum_IMU1 = pd.concat([acc2x_sum_IMU1, acc2y_sum_IMU1, acc2z_sum_IMU1], axis=1)
    acc2_sum_IMU1 = acc2_sum_IMU1.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    feature_df["acc1x_sum_IMU1"] = acc1_sum_IMU1["X"]
    feature_df["acc1y_sum_IMU1"] = acc1_sum_IMU1["Y"]
    feature_df["acc1z_sum_IMU1"] = acc1_sum_IMU1["Z"]
    
    feature_df["acc2x_sum_IMU1"] = acc2_sum_IMU1["X"]
    feature_df["acc2y_sum_IMU1"] = acc2_sum_IMU1["Y"]
    feature_df["acc2z_sum_IMU1"] = acc2_sum_IMU1["Z"]
#    
    ###*************************************************************************************
    
    
    ####IMU2
    xacc2 = acc_IMU2.iloc[:, 0:100]
    yacc2 = acc_IMU2.iloc[:, 100:200]
    zacc2 = acc_IMU2.iloc[:, 200:300]
    
    #sum of columns
    acc1x_sum_IMU2 = xacc2.sum(axis=1)
    xacc2_50 = xacc2.iloc[:, 0:50]
    acc2x_sum_IMU2 = xacc2_50.sum(axis=1)
    
    #sum of columns
    acc1y_sum_IMU2 = yacc2.sum(axis=1)
    yacc2_50 = yacc2.iloc[:, 0:50]
    acc2y_sum_IMU2 = yacc2_50.sum(axis=1)
    
    #sum of columns
    acc1z_sum_IMU2 = zacc2.sum(axis=1)
    zacc2_50 = zacc2.iloc[:, 0:50]
    acc2z_sum_IMU2 = zacc2_50.sum(axis=1)
    
    ###### Features
    acc1_sum_IMU2 = pd.concat([acc1x_sum_IMU2, acc1y_sum_IMU2, acc1z_sum_IMU2], axis=1)
    acc1_sum_IMU2 = acc1_sum_IMU2.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    acc2_sum_IMU2 = pd.concat([acc2x_sum_IMU2, acc2y_sum_IMU2, acc2z_sum_IMU2], axis=1)
    acc2_sum_IMU2 = acc2_sum_IMU2.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    feature_df["acc1x_sum_IMU2"] = acc1_sum_IMU2["X"]
    feature_df["acc1y_sum_IMU2"] = acc1_sum_IMU2["Y"]
    feature_df["acc1z_sum_IMU2"] = acc1_sum_IMU2["Z"]
    
    feature_df["acc2x_sum_IMU2"] = acc2_sum_IMU1["X"]
    feature_df["acc2y_sum_IMU2"] = acc2_sum_IMU2["Y"]
    feature_df["acc2z_sum_IMU2"] = acc2_sum_IMU2["Z"]
    ##***************************************************************************************************
    
    ####IMU3
    xacc3 = acc_IMU3.iloc[:, 0:100]
    yacc3 = acc_IMU3.iloc[:, 100:200]
    zacc3 = acc_IMU3.iloc[:, 200:300]
    
    #sum of columns
    acc1x_sum_IMU3 = xacc3.sum(axis=1)
    xacc3_50 = xacc3.iloc[:, 0:50]
    acc2x_sum_IMU3 = xacc3_50.sum(axis=1)
    
    #sum of columns
    acc1y_sum_IMU3 = yacc3.sum(axis=1)
    yacc3_50 = yacc3.iloc[:, 0:50]
    acc2y_sum_IMU3 = yacc3_50.sum(axis=1)
    
    #sum of columns
    acc1z_sum_IMU3 = zacc3.sum(axis=1)
    zacc3_50 = zacc3.iloc[:, 0:50]
    acc2z_sum_IMU3 = zacc3_50.sum(axis=1)
    
    ###### Features
    acc1_sum_IMU3 = pd.concat([acc1x_sum_IMU3, acc1y_sum_IMU3, acc1z_sum_IMU3], axis=1)
    acc1_sum_IMU3 = acc1_sum_IMU3.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    acc2_sum_IMU3 = pd.concat([acc2x_sum_IMU3, acc2y_sum_IMU3, acc2z_sum_IMU3], axis=1)
    acc2_sum_IMU3 = acc2_sum_IMU3.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    feature_df["acc1x_sum_IMU3"] = acc1_sum_IMU3["X"]
    feature_df["acc1y_sum_IMU3"] = acc1_sum_IMU3["Y"]
    feature_df["acc1z_sum_IMU3"] = acc1_sum_IMU3["Z"]
    
    feature_df["acc2x_sum_IMU3"] = acc2_sum_IMU3["X"]
    feature_df["acc2y_sum_IMU3"] = acc2_sum_IMU3["Y"]
    feature_df["acc2z_sum_IMU3"] = acc2_sum_IMU3["Z"]
    
    ###*********************************************************************************************
    
    ####IMU4
    xacc4 = acc_IMU4.iloc[:, 0:100]
    yacc4 = acc_IMU4.iloc[:, 100:200]
    zacc4 = acc_IMU4.iloc[:, 200:300]
    
    #sum of columns
    acc1x_sum_IMU4 = xacc4.sum(axis=1)
    xacc4_50 = xacc4.iloc[:, 0:50]
    acc2x_sum_IMU4 = xacc4_50.sum(axis=1)
    
    #sum of columns
    acc1y_sum_IMU4 = yacc4.sum(axis=1)
    yacc4_50 = yacc4.iloc[:, 0:50]
    acc2y_sum_IMU4 = yacc4_50.sum(axis=1)
    
    #sum of columns
    acc1z_sum_IMU4 = zacc4.sum(axis=1)
    zacc4_50 = zacc4.iloc[:, 0:50]
    acc2z_sum_IMU4 = zacc4_50.sum(axis=1)
    
    ###### Features
    acc1_sum_IMU4= pd.concat([acc1x_sum_IMU4, acc1y_sum_IMU4, acc1z_sum_IMU4], axis=1)
    acc1_sum_IMU4 = acc1_sum_IMU4.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    acc2_sum_IMU4 = pd.concat([acc2x_sum_IMU4, acc2y_sum_IMU4, acc2z_sum_IMU4], axis=1)
    acc2_sum_IMU4 = acc2_sum_IMU4.rename(index=str, columns = {0:'X', 1:'Y', 2:'Z'})
    
    feature_df["acc1x_sum_IMU4"] = acc1_sum_IMU4["X"]
    feature_df["acc1y_sum_IMU4"] = acc1_sum_IMU4["Y"]
    feature_df["acc1z_sum_IMU4"] = acc1_sum_IMU4["Z"]
    
    feature_df["acc2x_sum_IMU4"] = acc2_sum_IMU1["X"]
    feature_df["acc2y_sum_IMU4"] = acc2_sum_IMU4["Y"]
    feature_df["acc2z_sum_IMU4"] = acc2_sum_IMU4["Z"]
    
    feature_df.shape
    
    ########################calculate norm general function first feature ###########################################################################################

    
    def calculate_norm(x,y,z):
        """
        This function calculates the l2 norm of a vector
        """
        addition = np.square(x) + np.square(y) + np.square(z)
        norm2 = np.sqrt(addition)
        
        return norm2
        
        
    N_x_v = calculate_norm(xv1_3, yv1_3, zv1_3) + calculate_norm(xv1_4, yv1_4, zv1_4)
    N_y_v = calculate_norm(xv2_3, yv2_3, zv2_3) + calculate_norm(xv2_3, yv2_3, zv2_3)  
    
    N_x_a = calculate_norm(xa1_3, ya1_3, za1_3) + calculate_norm(xa1_4, ya1_4, za1_4)  
    N_y_a = calculate_norm(xa2_3, ya2_3, za2_3) + calculate_norm(xa2_3, ya2_3, za2_3)  
        

    
##################    
    feature_df["Norm-vel-diff-1-3-1-4(1)"] = N_x_v["Column1"].values
    feature_df["Norm-vel-diff-1-3-1-4(2)"] = N_x_v["Column2"].values
    feature_df["Norm-vel-diff-1-3-1-4(3)"] = N_x_v["Column3"].values
    feature_df["Norm-vel-diff-1-3-1-4(4)"] = N_x_v["Column4"].values
    feature_df["Norm-vel-diff-1-3-1-4(5)"] = N_x_v["Column5"].values
    feature_df["Norm-vel-diff-1-3-1-4(6)"] = N_x_v["Column6"].values
    feature_df["Norm-vel-diff-1-3-1-4(7)"] = N_x_v["Column7"].values
    feature_df["Norm-vel-diff-1-3-1-4(8)"] = N_x_v["Column8"].values
    feature_df["Norm-vel-diff-1-3-1-4(9)"] = N_x_v["Column9"].values
    feature_df["Norm-vel-diff-1-3-1-4(10)"] = N_x_v["Column10"].values
    
    
    feature_df["Norm-vel-diff-2-3-2-4(1)"] = N_y_v["Column1"].values
    feature_df["Norm-vel-diff-2-3-2-4(2)"] = N_y_v["Column2"].values
    feature_df["Norm-vel-diff-2-3-2-4(3)"] = N_y_v["Column3"].values
    feature_df["Norm-vel-diff-2-3-2-4(4)"] = N_y_v["Column4"].values
    feature_df["Norm-vel-diff-2-3-2-4(5)"] = N_y_v["Column5"].values
    feature_df["Norm-vel-diff-2-3-2-4(6)"] = N_y_v["Column6"].values
    feature_df["Norm-vel-diff-2-3-2-4(7)"] = N_y_v["Column7"].values
    feature_df["Norm-vel-diff-2-3-2-4(8)"] = N_y_v["Column8"].values
    feature_df["Norm-vel-diff-2-3-2-4(9)"] = N_y_v["Column9"].values
    feature_df["Norm-vel-diff-2-3-2-4(10)"] = N_y_v["Column10"].values
    
    
    feature_df["Norm-acc-diff-1-3-1-4(1)"] = N_x_a["Column1"].values
    feature_df["Norm-acc-diff-1-3-1-4(2)"] = N_x_a["Column2"].values
    feature_df["Norm-acc-diff-1-3-1-4(3)"] = N_x_a["Column3"].values
    feature_df["Norm-acc-diff-1-3-1-4(4)"] = N_x_a["Column4"].values
    feature_df["Norm-acc-diff-1-3-1-4(5)"] = N_x_a["Column5"].values
    feature_df["Norm-acc-diff-1-3-1-4(6)"] = N_x_a["Column6"].values
    feature_df["Norm-acc-diff-1-3-1-4(7)"] = N_x_a["Column7"].values
    feature_df["Norm-acc-diff-1-3-1-4(8)"] = N_x_a["Column8"].values
    feature_df["Norm-acc-diff-1-3-1-4(9)"] = N_x_a["Column9"].values
    feature_df["Norm-acc-diff-1-3-1-4(10)"] = N_x_a["Column10"].values
    
    
    feature_df["Norm-acc-diff-2-3-2-4(1)"] = N_y_a["Column1"].values
    feature_df["Norm-acc-diff-2-3-2-4(2)"] = N_y_a["Column2"].values
    feature_df["Norm-acc-diff-2-3-2-4(3)"] = N_y_a["Column3"].values
    feature_df["Norm-acc-diff-2-3-2-4(4)"] = N_y_a["Column4"].values
    feature_df["Norm-acc-diff-2-3-2-4(5)"] = N_y_a["Column5"].values
    feature_df["Norm-acc-diff-2-3-2-4(6)"] = N_y_a["Column6"].values
    feature_df["Norm-acc-diff-2-3-2-4(7)"] = N_y_a["Column7"].values
    feature_df["Norm-acc-diff-2-3-2-4(8)"] = N_y_a["Column8"].values
    feature_df["Norm-acc-diff-2-3-2-4(9)"] = N_y_a["Column9"].values
    feature_df["Norm-acc-diff-2-3-2-4(10)"] = N_y_a["Column10"].values
    

#############################calculating a cross w feature ######################################################    
    
    def rename_columns(m):
        """
        This function rename the name of the columns of a matrix
        """
        column_names = m.columns.values
        column_names[0] = 'X'
        column_names[1] = 'Y'
        column_names[2] = 'Z'
        m.columns = column_names
        return m
    
    
    def reshape_df(x,y,z):
        """
        This function is for reshaping the matrix or concanating the vectors into matrix
        """
        m1 = pd.concat([x["Column1"], y["Column1"], z["Column1"]], axis=1)
        m1 = rename_columns(m1)
        
        m2 = pd.concat([x["Column2"], y["Column2"], z["Column2"]], axis=1)
        m2 = rename_columns(m2)
        
        m3 = pd.concat([x["Column3"], y["Column3"], z["Column3"]], axis=1)
        m3 = rename_columns(m3)
        
        m4 = pd.concat([x["Column4"], y["Column4"], z["Column4"]], axis=1)
        m4 = rename_columns(m4)
        
        m5 = pd.concat([x["Column5"], y["Column5"], z["Column5"]], axis=1)
        m5 = rename_columns(m5) 
        
        m6 = pd.concat([x["Column6"], y["Column6"], z["Column6"]], axis=1)
        m6 = rename_columns(m6)
        
        m7 = pd.concat([x["Column7"], y["Column7"], z["Column7"]], axis=1)
        m7 = rename_columns(m7)
        
        m8 = pd.concat([x["Column8"], y["Column8"], z["Column8"]], axis=1)
        m8 = rename_columns(m8)
        
        m9 = pd.concat([x["Column9"], y["Column9"], z["Column9"]], axis=1)
        m9 = rename_columns(m9)
        
        m10 = pd.concat([x["Column10"], y["Column10"], z["Column10"]], axis=1)
        m10 = rename_columns(m10)

        result = pd.concat([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], axis = 0)
        
        return result
    
    
    v1 = reshape_df(xv1, yv1, zv1)
    a1 = reshape_df(xa1, ya1, za1)
    
    r1 = np.cross(v1, a1)
    
    norm_r1 = calculate_norm(r1[:,0], r1[:,1], r1[:,2])
    
    r1_splitted = np.split(norm_r1,10)
    
    
    
    v2 = reshape_df(xv2, yv2, zv2)
    a2 = reshape_df(xa2, ya2, za2)
    
    r2 = np.cross(v2, a2)
    
    norm_r2 = calculate_norm(r2[:,0], r2[:,1], r2[:,2])
    
    r2_splitted = np.split(norm_r2,10)
    
    
    
    
    v3 = reshape_df(xv3, yv3, zv3)
    a3 = reshape_df(xa3, ya3, za3)
    
    r3 = np.cross(v3, a3)
    
    norm_r3 = calculate_norm(r3[:,0], r3[:,1], r3[:,2])
    
    r3_splitted = np.split(norm_r3,10)
    
    
    
    v4 = reshape_df(xv4, yv4, zv4)
    a4 = reshape_df(xa4, ya4, za4)
    
    r4 = np.cross(v4, a4)
    
    norm_r4 = calculate_norm(r4[:,0], r4[:,1], r4[:,2])
    
    r4_splitted = np.split(norm_r4,10)
    
    
    
    feature_df["Norm-IMU1-1-W^-1a"] = r1_splitted[0]
    feature_df["Norm-IMU1-2-W^-1a"] = r1_splitted[1]
    feature_df["Norm-IMU1-3-W^-1a"] = r1_splitted[2]
    feature_df["Norm-IMU1-4-W^-1a"] = r1_splitted[3]
    feature_df["Norm-IMU1-5-W^-1a"] = r1_splitted[4]
    feature_df["Norm-IMU1-6-W^-1a"] = r1_splitted[5]
    feature_df["Norm-IMU1-7-W^-1a"] = r1_splitted[6]
    feature_df["Norm-IMU1-8-W^-1a"] = r1_splitted[7]
    feature_df["Norm-IMU1-9-W^-1a"] = r1_splitted[8]
    feature_df["Norm-IMU1-10-W^-1a"] = r1_splitted[9]
    
    
    feature_df["Norm-IMU2-1-W^-1a"] = r2_splitted[0][0]
    feature_df["Norm-IMU2-2-W^-1a"] = r2_splitted[1]
    feature_df["Norm-IMU2-3-W^-1a"] = r2_splitted[2]
    feature_df["Norm-IMU2-4-W^-1a"] = r2_splitted[3]
    feature_df["Norm-IMU2-5-W^-1a"] = r2_splitted[4]
    feature_df["Norm-IMU2-6-W^-1a"] = r2_splitted[5]
    feature_df["Norm-IMU2-7-W^-1a"] = r2_splitted[6]
    feature_df["Norm-IMU2-8-W^-1a"] = r2_splitted[7]
    feature_df["Norm-IMU2-9-W^-1a"] = r2_splitted[8]
    feature_df["Norm-IMU2-10-W^-1a"] = r2_splitted[9]
    
    
    feature_df["Norm-IMU3-1-W^-1a"] = r3_splitted[0]
    feature_df["Norm-IMU3-2-W^-1a"] = r3_splitted[1]
    feature_df["Norm-IMU3-3-W^-1a"] = r3_splitted[2]
    feature_df["Norm-IMU3-4-W^-1a"] = r3_splitted[3]
    feature_df["Norm-IMU3-5-W^-1a"] = r3_splitted[4]
    feature_df["Norm-IMU3-6-W^-1a"] = r3_splitted[5]
    feature_df["Norm-IMU3-7-W^-1a"] = r3_splitted[6]
    feature_df["Norm-IMU3-8-W^-1a"] = r3_splitted[7]
    feature_df["Norm-IMU3-9-W^-1a"] = r3_splitted[8]
    feature_df["Norm-IMU3-10-W^-1a"] = r3_splitted[9]
    
    feature_df["Norm-IMU4-1-W^-1a"] = r4_splitted[0]
    feature_df["Norm-IMU4-2-W^-1a"] = r4_splitted[1]
    feature_df["Norm-IMU4-3-W^-1a"] = r4_splitted[2]
    feature_df["Norm-IMU4-4-W^-1a"] = r4_splitted[3]
    feature_df["Norm-IMU4-5-W^-1a"] = r4_splitted[4]
    feature_df["Norm-IMU4-6-W^-1a"] = r4_splitted[5]
    feature_df["Norm-IMU4-7-W^-1a"] = r4_splitted[6]
    feature_df["Norm-IMU4-8-W^-1a"] = r4_splitted[7]
    feature_df["Norm-IMU4-9-W^-1a"] = r4_splitted[8]
    feature_df["Norm-IMU4-10-W^-1a"] = r4_splitted[9]

    
    
    feature_df.to_csv(path_or_buf =
                    'P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\New feature dataset based on acceleration whole data\\Feature_Engineered_dataset_whole_feature1-3.csv', 
                    sep=',')
    
    IMU_class = IMU_class.rename(index=str, columns = {0:'Class'})
    IMU_class.to_csv(path_or_buf =
                    'P:\\My Documents\\Thesis\\Feature Engineering result new dataset(vector operators)\\Class_dataset_whole.csv', sep=',')
    
    
