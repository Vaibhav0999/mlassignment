# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:13:16 2022

@author: Admin
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd

import streamlit as st

################################
# model
################################

# load model
#print("\n*** Load Model ***")
import pickle
filename = 'E:/ml assignment/heart.pkl'
model = pickle.load(open(filename, 'rb'))
#print(model)
#print("Done ...")

# load vars
#print("\n*** Load Vars ***")
filename = 'E:/ml assignment//heart-vars.pkl'
dVars = pickle.load(open(filename, 'rb'))
#print(dVars)
clsVars = dVars['clsVars'] 
allCols = dVars['allCols']

#print("Done ...")

################################
# predict
#######N#######################

def getPredict(dfp):
    global clsVars, allCols, Le
    
    # split into data & outcome
    X_pred = dfp[allCols].values
    #y_pred = dfp[clsVars].values
    # predict from model
    p_pred = model.predict(X_pred)
    # update data frame
    dfp['Predict'] = p_pred
  

    return (dfp)

 
########################################################
# headers
########################################################
# title
st.title("Heart Disease Possiblility Prediction")

# title
st.sidebar.title("Patient Data")

# column = df["glucose"]
# max_value = column.min()
# max_value
vage = st.sidebar.slider("Age", 30, 70, 50)
vtotChol = st.sidebar.slider('Cholesterol' , 0, 700, 150)
vsysBP = st.sidebar.slider('SystolicBP',0.0, 225.0, 115.0)
vdiaBP = st.sidebar.slider('DiastolicBP' , 0.0, 135.0, 35.5)
vBMI = st.sidebar.slider('BMI' , 0.0, 60.0, 30.0)
vheartRate = st.sidebar.slider('HeartRate' ,0, 145,40)
vglucose = st.sidebar.slider('Glucose' ,0, 400, 40)

# submit
if(st.sidebar.button("Submit")):
    def user_report():
        user_report_data = {
                'age': vage,
                'totChol' : vtotChol,
                'sysBP': vsysBP,
                'diaBP' : vdiaBP,
                'BMI' : vBMI,
                'heartRate': vheartRate,
                'glucose': vglucose}
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data
        
    # PATIENT DATA
    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)
    
     # create data frame for predict
    dfp = pd.DataFrame(user_data, index=[0])
    dfp = getPredict(dfp)
    
    # OUTPUT
    st.subheader('Your Report: ')
    output = ''
    if dfp['Predict'][0]:
        output = 'You will have a 10 year risk of developing a cardio vascular disease'
        
    else:
        output = 'You are Fine, but need to take care'
    st.title(output)
    # reset    
    st.button("Reset")