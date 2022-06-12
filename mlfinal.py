# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 18:06:24 2022

@author: shilp
"""

import streamlit as st
import pandas as pd
import math
import joblib
import numpy as np
import pickle as pkl 
st.title("ASTEROID DIAMETER PREDICTION")
st.write("Measuring diameter of asteroids is very hard and complicated process 😫 . Machine learning comes to rescue i just need some parameters of asteroids and i will predict diameter of asteroid for you !!! 😍 .I know most of you are not scientists don't worry i will provide you a range for all the features if you don't know you can choose from it .")
#Eccentricity
e=st.number_input("Eccentricity of elliptical path of asteroid . Range(0.00-0.98)")
#Inclination with respect to x-y plane
i=st.number_input("Inclination with respect to x-y plane. Range(0.044-151.81)")
#Longitude of ascending node 
om=st.number_input("Longitude of ascending node. Range(0.0007-359.99)")
#Perihelion distance 
q=st.number_input("Perihelion distance . Range(0.08-40.46)")
#Aphelion distance
ad=st.number_input("Aphelion distance . Range(0.99-772.20)")
#Data Arc Span
data_arc=st.number_input("Data Arc Span . Range(1.0-72684.0)")
#Orbital condition code
condition_Code = st.selectbox('Condition code',(0,1,2,3,4,5,6,7,8,9))
#Absolute magnitude parameter
H=st.number_input("Absolute magnitude parameter . Range(3.6-29.9)")
#NEAR EARTH OBJECTS
neo = st.selectbox('Asteroid is near-earth object or not ?',('Y','N'))
#Potentially hazardous object
pha = st.selectbox('Asteroid is potentially hazardous object or not ?',('Y','N'))
#Albedo
albedo=st.number_input("Albedo .  Range(0.001-1.0)")
#Moid 
moid = st.number_input("Moid .  Range(0.0003-39.50)")
#Class
clas = st.selectbox('Class',('MBA','OMB','TJN','IMB','APO','MCA','AMO','ATE','CEN','TNO','AST'))

#Creating a temp_dictionary
temp_dictionary={'a':[5.0],'e':[],'i':[],'om':[],'w':[5.0],'q':[],'ad':[],'per_y':[5.0],'data_arc':[],'n_obs_used':[5],'H':[],'neo':[],'pha':[],'albedo':[],'moid':[],'class':[],'n':[5.0],'per':[5.0],'ma':[5.0],'estimate_diameter':[],'condition_code':[]}
temp_dictionary['e'].append(e)
temp_dictionary['i'].append(i)
temp_dictionary['om'].append(om)
temp_dictionary['q'].append(q)
temp_dictionary['ad'].append(ad)
temp_dictionary['data_arc'].append(data_arc)
temp_dictionary['condition_code'].append(condition_Code)
temp_dictionary['H'].append(H)
temp_dictionary['neo'].append(neo)
temp_dictionary['pha'].append(pha)
temp_dictionary['albedo'].append(albedo)
temp_dictionary['moid'].append(moid)
temp_dictionary['class'].append(clas)
if st.button('Calculate Diameter'):
     r=(3.1236-(0.5*math.log10(albedo))-(0.2*H))
     d=math.pow(10,r)
     temp_dictionary['estimate_diameter'].append(d)
     temp_df=pd.DataFrame(temp_dictionary)
     X_train_temp=temp_df.loc[:,['condition_code','neo','pha','class']]
     X_train_numerical=temp_df.drop(['condition_code','neo','pha','class'],axis=1)
     sc=joblib.load('final_standardscaler.joblib')
     X_train_numerical_scaler=sc.transform(X_train_numerical)
     X_train_numerical_scaler=pd.DataFrame(X_train_numerical_scaler)
     X_train_numerical_scaler.columns=['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used', 'H', 'albedo', 'moid', 'n', 'per', 'ma', 'estimate_diameter']
     X_train_temp={'condition_code':[], 'neo_N':[], 'neo_Y':[], 'pha_N':[], 'pha_Y':[], 'class_AMO':[], 'class_APO':[], 'class_AST':[], 'class_ATE':[], 'class_CEN':[], 'class_IMB':[], 'class_MBA':[], 'class_MCA':[], 'class_OMB':[], 'class_TJN':[], 'class_TNO':[]}
     #for condition code
     X_train_temp['condition_code'].append(float(condition_Code))
     #For Neo
     if neo=='Y' :
         X_train_temp['neo_N'].append(0)
         X_train_temp['neo_Y'].append(1)
     elif neo=='N' :
         X_train_temp['neo_N'].append(1)
         X_train_temp['neo_Y'].append(0)
     #For pha 
     if pha=='Y' :
         X_train_temp['pha_N'].append(0)
         X_train_temp['pha_Y'].append(1)
     elif pha=='N' :
         X_train_temp['pha_N'].append(1)
         X_train_temp['pha_Y'].append(0)
    #For class 
     if clas=='AMO' :
         X_train_temp['class_AMO'].append(1)
         X_train_temp['class_APO'].append(0)
         X_train_temp['class_AST'].append(0)
         X_train_temp['class_ATE'].append(0)
         X_train_temp['class_CEN'].append(0)
         X_train_temp['class_IMB'].append(0)
         X_train_temp['class_MBA'].append(0)
         X_train_temp['class_MCA'].append(0)
         X_train_temp['class_OMB'].append(0)
         X_train_temp['class_TJN'].append(0)
         X_train_temp['class_TNO'].append(0)
     elif clas=='APO' :
         X_train_temp['class_AMO'].append(0)
         X_train_temp['class_APO'].append(1)
         X_train_temp['class_AST'].append(0)
         X_train_temp['class_ATE'].append(0)
         X_train_temp['class_CEN'].append(0)
         X_train_temp['class_IMB'].append(0)
         X_train_temp['class_MBA'].append(0)
         X_train_temp['class_MCA'].append(0)
         X_train_temp['class_OMB'].append(0)
         X_train_temp['class_TJN'].append(0)
         X_train_temp['class_TNO'].append(0)
     elif clas=='AST' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(1)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='ATE' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(1)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='CEN' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(1)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='IMB' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(1)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='MBA' :
         X_train_temp['class_AMO'].append(0)
         X_train_temp['class_APO'].append(0)
         X_train_temp['class_AST'].append(0)
         X_train_temp['class_ATE'].append(0)
         X_train_temp['class_CEN'].append(0)
         X_train_temp['class_IMB'].append(0)
         X_train_temp['class_MBA'].append(1)
         X_train_temp['class_MCA'].append(0)
         X_train_temp['class_OMB'].append(0)
         X_train_temp['class_TJN'].append(0)
         X_train_temp['class_TNO'].append(0)
     elif clas=='MCA' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(1)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='OMB' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(1)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(0)
     elif clas=='TJN' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(1)
        X_train_temp['class_TNO'].append(0)
     elif clas=='TNO' :
        X_train_temp['class_AMO'].append(0)
        X_train_temp['class_APO'].append(0)
        X_train_temp['class_AST'].append(0)
        X_train_temp['class_ATE'].append(0)
        X_train_temp['class_CEN'].append(0)
        X_train_temp['class_IMB'].append(0)
        X_train_temp['class_MBA'].append(0)
        X_train_temp['class_MCA'].append(0)
        X_train_temp['class_OMB'].append(0)
        X_train_temp['class_TJN'].append(0)
        X_train_temp['class_TNO'].append(1)
     X_train_temp=pd.DataFrame(X_train_temp)
     X_train=pd.DataFrame(np.concatenate([X_train_numerical_scaler.values,X_train_temp.values],axis=1))
     X_train.columns=['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used', 'H', 'albedo', 'moid', 'n', 'per', 'ma', 'estimate_diameter', 'condition_code', 'neo_N', 'neo_Y', 'pha_N', 'pha_Y', 'class_AMO', 'class_APO', 'class_AST', 'class_ATE', 'class_CEN', 'class_IMB', 'class_MBA', 'class_MCA', 'class_OMB', 'class_TJN', 'class_TNO']
     X_train=X_train.drop(['ma','w','a','per_y','per','n','n_obs_used'],axis=1)
     final_model=pkl.load(open('final_model.pkl', 'rb'))
     pred=final_model.predict(X_train.values)[0][0]
     st.title("Diameter of asteroid is {} km".format(pred))




