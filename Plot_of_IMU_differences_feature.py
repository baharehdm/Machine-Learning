# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:37:35 2018

@author: darvishm
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import os

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
    
    x = np.array(X_train)
    
    
    print(x.shape)
    print(y.shape)
    

    
    def column_list(size):
        
        return ["Column" + str(i) for i in range(1,size + 1)]


        
    
    IMU_data = pd.DataFrame.from_records(x, columns = column_list(240))
    
    IMU_class = pd.DataFrame.from_records(y)
    
    
    ############### IMU1-IMU2 velocity
    xv1 = IMU_data.iloc[:, 0:10]
    xv2 = IMU_data.iloc[:, 60:70]
    
    """change the column name because it needs to be the same column names on both df that we want to subtract"""
    
    xv2.columns = column_list(10)
    
    
    xv1_2 = xv1.sub(xv2, fill_value=0)
    
    yv1 = IMU_data.iloc[:, 10:20]
    yv2 = IMU_data.iloc[:, 70:80]
    
    yv2.columns = column_list(10)
    yv1.columns = column_list(10)
    
    yv1_2 = yv1.sub(yv2, fill_value=0)
    
    zv1 = IMU_data.iloc[:, 20:30]
    zv2 = IMU_data.iloc[:, 80:90]
    zv2.columns = column_list(10)
    zv1.columns = column_list(10)
    
    zv1_2 = zv1.sub(zv2, fill_value=0)
    
    ################ IMU3-IMU4 velocity
    xv3 = IMU_data.iloc[:, 120:130]
    xv4 = IMU_data.iloc[:, 180:190]
    xv3.columns = column_list(10)
    xv4.columns = column_list(10)
    
    xv3_4 = xv3.sub(xv4, fill_value=0)
    
    
    
    yv3 = IMU_data.iloc[:, 130:140]
    yv4 = IMU_data.iloc[:, 190:200]
    yv3.columns = column_list(10)
    yv4.columns = column_list(10)
    
    yv3_4 = yv3.sub(yv4, fill_value=0)
    
    
    zv3 = IMU_data.iloc[:, 140:150]
    zv4 = IMU_data.iloc[:, 200:210]
    zv3.columns = column_list(10)
    zv4.columns = column_list(10)
    
    zv3_4 = zv3.sub(zv4, fill_value=0)
    
    
    
    ############### IMU1-IMU2 acceleration
    xa1 = IMU_data.iloc[:, 30:40]
    xa2 = IMU_data.iloc[:, 90:100]
    xa1.columns = column_list(10)
    xa2.columns = column_list(10)
    
    
    xa1_2 = xa1.sub(xa2, fill_value=0)
    
    ya1 = IMU_data.iloc[:, 40:50]
    ya2 = IMU_data.iloc[:, 100:110]
    ya2.columns = column_list(10)
    ya1.columns = column_list(10)
    
    ya1_2 = ya1.sub(ya2, fill_value=0)
    
    za1 = IMU_data.iloc[:, 50:60]
    za2 = IMU_data.iloc[:, 110:120]
    za2.columns = column_list(10)
    za1.columns = column_list(10)
    
    za1_2 = za1.sub(za2, fill_value=0)
    
    ################ IMU3-IMU4 acceleration
    xa3 = IMU_data.iloc[:, 150:160]
    xa4 = IMU_data.iloc[:, 210:220]
    xa3.columns = column_list(10)
    xa4.columns = column_list(10)
    
    xa3_4 = xa3.sub(xa4, fill_value=0)
    
    
    
    ya3 = IMU_data.iloc[:, 160:170]
    ya4 = IMU_data.iloc[:, 220:230]
    ya3.columns = column_list(10)
    ya4.columns = column_list(10)
    
    ya3_4 = ya3.sub(ya4, fill_value=0)
    
    
    za3 = IMU_data.iloc[:, 170:180]
    za4 = IMU_data.iloc[:, 230:240]
    za3.columns = column_list(10)
    za4.columns = column_list(10)
    
    za3_4 = za3.sub(za4, fill_value=0)
    
    
    
    #######################################angular velocities differences###########################################
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
    
    ###################################linear accelations differences####################################################
    
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
    
    
    
    def joint(label):
        """
        This function gives the label name correspond to each class
        """
        if label == 0:
            return 'rigid'
        elif label == 1:
            return 'revolute'
        elif label == 2:
            return 'prismatic'
        elif label == 3:
            return 'corrupted'
        elif label == 4:
            return 'random'
        
        
        
        
    
    import random
    
    label_dict = {}
    
    """ Generating 100 random number between 0 and 40000 and making a dictionary consisting of 
    the row of data as a key and the label word(ex: rigid) as a value."""
    for i in range(0,100):
        n = random.randint(0, len(IMU_data))
        label = joint(IMU_class.loc[n,0])
        label_dict[n] = label
        
   
    rigid_list = []
    revolute_list = []
    prismatic_list = []
    corrupted_list = []
    random_list = []
    
    """ making list for each label """
    for item in label_dict:
        if label_dict[item] == 'rigid':
            rigid_list.append(item)
        elif label_dict[item] == 'revolute':
            revolute_list.append(item)
        elif label_dict[item] == 'prismatic':
            prismatic_list.append(item)
        elif label_dict[item] == 'corrupted':
            corrupted_list.append(item)
        elif label_dict[item] == 'random':
            random_list.append(item)
            
    
    
    import numpy as np
    
    def calculate_variance(label_list, col_axis):
        """ This function calculate the variance of each column on a specific axis """
        len_label = len(label_list)
        if len_label != 0:
            for i in range(0,len_label):
                var = np.round(np.var(col_axis.iloc[label_list[i],:]), 4)
        
        return var
    
    
    def calculate_mean(label_list, col_axis):
        """ This function calculate the mean of each column on a specific axis """
        len_label = len(label_list)
        if len_label != 0:
            for i in range(0,len_label):
                mean = np.round(np.mean(col_axis.iloc[label_list[i],:]), 4)
        
        return mean
            
            
            
    
    
    def plot_differences_feature_new(label_list, title, axis1, axis2, 
                                     col1_3_axis1, col2_4_axis1, col1_3_axis2, 
                                     col2_4_axis2, filename):
        
        """
        This function represents the differences of IMUs(1-2,3-4)
        in a scatter plot and shows each row of the data in a 
        different color
        
        param: label_list: list of each label (rigid, revolute, prismatic, random 
                                               or corrupted)
        param: title: title of the plot
        param: axis1 : first axis
        param: axis2: second axis
        param: col1_3_axis1: diffence of IMU 1-3 Dataframe for the first axis
        param: col2_4_axis1: diffence of IMU 2-4 Dataframe for the second axis
        param: col1_3_axis2: diffence of IMU 1-3 Dataframe for the first axis
        param: col2_4_axis2: diffence of IMU 2-4 Dataframe for the second axis
        param: filename: file name for saving the plot
        
        """

        colors_lst = ['red', 'yellow','orange' , 'green', 'cyan', 'blue' ,
                      'purple' ,'pink', 'navy','olive' ,'peru',  'tomato', 
                      'chocolate' ,'tan', 'khaki', 'darkkhaki', 'yellowgreen', 
                      'c','deepskyblue' ,'navy' ,'violet', 'hotpink', 'rosybrown', 
                      'black', 'brown', 'wheat', 'lime']
        
        
        var1_2_x = calculate_variance(label_list, col1_3_axis1)
        mean1_2_x = calculate_mean(label_list, col1_3_axis1)
        var3_4_x = calculate_variance(label_list, col2_4_axis1)
        mean3_4_x = calculate_mean(label_list, col2_4_axis1)
        
        var1_2_y = calculate_variance(label_list, col1_3_axis2)
        mean1_2_y = calculate_mean(label_list, col1_3_axis2)
        var3_4_y = calculate_variance(label_list, col2_4_axis2)
        mean3_4_y = calculate_mean(label_list, col2_4_axis2)
        
        len_label = len(label_list)
        if len_label != 0:
            plt.style.use('ggplot')
            fig = plt.figure()
            fig.suptitle(title, fontsize=14)
            
            plt.xlabel(axis1 + ' 1-3('+ str(mean1_2_x) + ', ' + str(var1_2_x)  + ')' + 
                        ' 2-4(' + str(mean3_4_x) + ', ' + str(var3_4_x) +  ')'  
                        ,fontsize=10)
        
            plt.ylabel(axis2 + ' 1-3(' + str(mean1_2_y) + ', ' + str(var1_2_y) + ')' + 
                       ' 2-4(' + str(mean3_4_y) + ', ' + str(var3_4_y) + ')'
                       , fontsize=10)
            
            
            ax1 = fig.add_subplot(111)
            for i in range(0,5):
                ax1.scatter(col1_3_axis1.iloc[label_list[i],:], 
                            col1_3_axis2.iloc[label_list[i],:], 
                            c = colors_lst[i]  , marker = 'x' )
                
                ax1.scatter(col2_4_axis1.iloc[label_list[i],:], 
                            col2_4_axis2.iloc[label_list[i],:], 
                            c = colors_lst[i] , marker = 'o' )
            
            
            leg = plt.gca().legend(('IMU1 - IMU3','IMU2 - IMU4'))
            leg.legendHandles[0].set_color('black')
            leg.legendHandles[1].set_color('black')

            path = "P:\\Plot of differences for new dataset1-3-2-4\\"
            fig.savefig(os.path.join(path, filename))
            
            
            
    ############################# calling the plot differences function for 1-3, 2-4 ##############################################################      
    #############################################each joint in one plot for accelarion for x,y ###############################################
    
    plot_differences_feature_new(rigid_list, 'Difference between linear accelerations for rigid joints', 'X', 'Y',
                      xa1_3, xa2_4, ya1_3, ya2_4, 'Plot_rigid_differencesxy.pdf' )
    
    plot_differences_feature_new(revolute_list, 'Difference between linear accelerations for revolute joints', 'X', 'Y',
                      xa1_3, xa2_4, ya1_3, ya2_4, 'Plot_rev_differencesxy.pdf' )
    
    plot_differences_feature_new(prismatic_list, 'Difference between linear accelerations for prismatic joints', 'X', 'Y',
                      xa1_3, xa2_4, ya1_3, ya2_4, 'Plot_pris_differencesxy.pdf' )
    
    plot_differences_feature_new(corrupted_list, 'Difference between linear accelerations for corrupted', 'X', 'Y',
                      xa1_3, xa2_4, ya1_3, ya2_4, 'Plot_corr_differencesxy.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between linear accelerations for random', 'X', 'Y',
                      xa1_3, xa2_4, ya1_3, ya2_4, 'Plot_random_differencesxy.pdf' )
    
    
     ############################################################## each joint in one plot for accelarion for x,z##############################
    plot_differences_feature_new(rigid_list, 'Difference between linear accelerations for rigid joints', 'X', 'Z',
                      xa1_3, xa2_4, za1_3, za2_4, 'Plot_rigid_differencesxz.pdf'  )
    
    plot_differences_feature_new(revolute_list, 'Difference between linear accelerations for revolute joints',  'X', 'Z',
                      xa1_3, xa2_4, za1_3, za2_4, 'Plot_rev_differencesxz.pdf' )
    
    plot_differences_feature_new(prismatic_list, 'Difference between linear accelerations for prismatic joints',  'X', 'Z',
                      xa1_3, xa2_4, za1_3, za2_4, 'Plot_pris_differencesxz.pdf' )
    
    plot_differences_feature_new(corrupted_list, 'Difference between linear accelerations for corrupted',  'X', 'Z',
                      xa1_3, xa2_4, za1_3, za2_4, 'Plot_corr_differencesxz.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between linear accelerations for random',  'X', 'Z',
                      xa1_3, xa2_4, za1_3, za2_4, 'Plot_random_differencesxz.pdf' )
    
    
     ###################################################each joint in one plot for accelarion for y,z###############################################################################
    plot_differences_feature_new(rigid_list, 'Difference between linear accelerations for rigid joints', 'Y', 'Z',
                      ya1_3, ya2_4, za1_3, za2_4, 'Plot_rigid_differencesyz.pdf' )
    
    plot_differences_feature_new(revolute_list, 'Difference between linear accelerations for revolute joints',  'Y', 'Z',
                      ya1_3, ya2_4, za1_3, za2_4, 'Plot_rev_differencesyz.pdf' )
    
    plot_differences_feature_new(prismatic_list, 'Difference between linear accelerations for prismatic joints',  'Y', 'Z',
                      ya1_3, ya2_4, za1_3, za2_4, 'Plot_pris_differencesyz.pdf' )
    
    plot_differences_feature_new(corrupted_list, 'Difference between linear accelerations for corrupted',  'Y', 'Z',
                      ya1_3, ya2_4, za1_3, za2_4, 'Plot_corr_differencesyz.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between linear accelerations for random',  'Y', 'Z',
                      ya1_3, ya2_4, za1_3, za2_4, 'Plot_random_differencesyz.pdf' )
    
    ############################################################################################################################################
    
    
    #*********************************************************************VELOCITY *************************************************************
    
     ###################################################each joint in one plot for velocity for x,y###############################################################################
    plot_differences_feature_new(rigid_list, 'Difference between angular velocities for rigid joints', 'X', 'Y',
                      xv1_3, xv2_4, yv1_3, yv2_4, 'Plot_rigid_differencesxy.pdf' )
    
    plot_differences_feature_new(revolute_list, 'Difference between angular velocities for revolute joints', 'X', 'Y',
                      xv1_3, xv2_4, yv1_3, yv2_4, 'Plot_rev_differencesxy.pdf')
    
    plot_differences_feature_new(prismatic_list, 'Difference between angular velocities for prismatic joints', 'X', 'Y',
                      xv1_3, xv2_4, yv1_3, yv2_4, 'Plot_pris_differencesxy.pdf')
    
    plot_differences_feature_new(corrupted_list, 'Difference between angular velocities for corrupted', 'X', 'Y',
                      xv1_3, xv2_4, yv1_3, yv2_4, 'Plot_corr_differencesxy.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between angular velocities for random', 'X', 'Y',
                      xv1_3, xv2_4, yv1_3, yv2_4, 'Plot_random_differencesxy.pdf' )
    
     #####################################################each joint in one plot for velocity for x,z###############################################################################
    plot_differences_feature_new(rigid_list, 'Difference between angular velocities for rigid joints', 'X', 'Z',
                      xv1_3, xv2_4, zv1_3, zv2_4, 'Plot_rigid_differencesxz.pdf' )
    
    plot_differences_feature_new(revolute_list, 'Difference between angular velocities for revolute joints',  'X', 'Z',
                      xv1_3, xv2_4, zv1_3, zv2_4, 'Plot_rev_differencesxz.pdf' )
    
    plot_differences_feature_new(prismatic_list, 'Difference between angular velocities for prismatic joints',  'X', 'Z',
                      xv1_3, xv2_4, zv1_3, zv2_4, 'Plot_pris_differencesxz.pdf' )
    
    plot_differences_feature_new(corrupted_list, 'Difference between angular velocities for corrupted',  'X', 'Z',
                      xv1_3, xv2_4, zv1_3, zv2_4, 'Plot_cor_differencesxz.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between angular velocities for random',  'X', 'Z',
                      xv1_3, xv2_4, zv1_3, zv2_4, 'Plot_random_differencesxz.pdf' )
    
    
    
    
   ###################################################each joint in one plot for velocity for y,z###############################################################################
    plot_differences_feature_new(rigid_list, 'Difference between angular velocities for rigid joints', 'Y', 'Z',
                      yv1_3, yv2_4, zv1_3, zv2_4, 'Plot_rigid_differencesyz.pdf' )
    
    plot_differences_feature_new(revolute_list, 'Difference between angular velocities for revolute joints',  'Y', 'Z',
                      yv1_3, yv2_4, zv1_3, zv2_4, 'Plot_rev_differencesyz.pdf' )
    
    plot_differences_feature_new(prismatic_list, 'Difference between angular velocities for prismatic joints',  'Y', 'Z',
                      yv1_3, yv2_4, zv1_3, zv2_4, 'Plot_pris_differencesyz.pdf' )
    
    plot_differences_feature_new(corrupted_list, 'Difference between angular velocities for corrupted',  'Y', 'Z',
                      yv1_3, yv2_4, zv1_3, zv2_4, 'Plot_corr_differencesyz.pdf' )
    
    plot_differences_feature_new(random_list, 'Difference between angular velocities for random',  'Y', 'Z',
                      yv1_3, yv2_4, zv1_3, zv2_4, 'Plot_random_differencesyz.pdf' )
    