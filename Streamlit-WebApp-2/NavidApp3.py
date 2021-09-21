import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import pickle 
import altair as alt

st.header('Hello, *World!* My Name Is Navid :sunglasses:')

#summary for input data

st.subheader('**Summary For Input Data**')
st.write( '** Credit Score**: Enter a 3-Digit Number Or Greater ** Such as: ** 550,2024,...')
st.write( '** Geography **: Enter 0,1 Or 2 ** Where ** 0: Germany, 1: Spain and 2: France')
st.write( '** Gender **: Enter 0 Or 1 ** Where ** 0: Female and 1: Male')
st.write( '** Age **: Enter a Number ** Such as: ** 30,45,92,...')
st.write( '** Tenure **: Enter a Number between 0 and 10 ** Such as: ** 2,8,10,...')
st.write( '** Balance **: Enter a Number between 0 and 250898.00 ** Such as: ** 159660.80,...')
st.write( '** Num Of Products **: Enter a Number between 1 and 4')
st.write( '** HasCrCard **: Enter 0 Or 1 ** Where ** 0: does not have and 1: have')
st.write( '** IsActiveMember **: Enter 0 Or 1 ** Where ** 0: not Active and 1: Active')
st.write( '** EstimatedSalary **: Enter a Number Or Greater ** Such as: ** 112542.58, 199992.00,12,...')

#Header-IMG
st.image('customer-churn-analysis-cover.png', width = 670) 
#Song
st.header('Ok,You can playing the Song for More Relaxation :sunglasses:')
audio_file = open('Somos.ogg', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

#Lets Goo
st.title('**Customer Churn Prediction App**')

#Load Catboost Model              
pickle_in= open('Catboost.pkl', 'rb')
classifier = pickle.load(pickle_in) 

#Create function for prediction and prediction_Prob
def predict_churn(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary):
    prediction=    classifier.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])
    print(prediction)
    return prediction

def predict_churn_proba(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary): 
    prediction_Prob=    classifier.predict_proba([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])
    print(prediction_Prob)
    return prediction_Prob
    
#Create pie Function For Ploting Probability
def make_pie(sizes, text, colors, labels):
    col = [[i / 255. for i in c] for c in colors]
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.45
    kwargs = dict(colors=col, startangle=180)
    outside, _ = ax.pie(sizes, radius=1, pctdistance=1 - width / 2, labels=labels, **kwargs)
    plt.setp(outside, width=width, edgecolor='white')
    kwargs = dict(size=15, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    ax.set_facecolor('#e6eaf1')
 
### Create Input Var
CreditScore = st.number_input(label='Credit Score',min_value=0,value=619)
Geography = st.radio("Geography",[0,1,2])
Gender = st.radio("Gender",[0,1])
Age = st.number_input(label="Age",min_value=0,value=40)
Tenure = st.number_input(label="Tenure",min_value=0,value=2)
Balance = st.number_input(label="Balance",min_value=0.00,value=159660.80)
NumOfProducts = st.radio("NumOfProducts",[0,1,2,3,4])
HasCrCard = st.radio("HasCrCard",[0,1])
IsActiveMember = st.radio("Active Member",[0,1])
EstimatedSalary = st.number_input(label="Estimated Salary",min_value=0.00,value=103921.57)
result =""

data = {'CreditScore': CreditScore,'Geography':Geography,'Gender':Gender,'Age': Age,'Tenure':Tenure,'Balance': Balance,'NumOfProducts':NumOfProducts,'HasCrCard':HasCrCard,'IsActiveMember':IsActiveMember,'EstimatedSalary':EstimatedSalary}            
features = pd.DataFrame(data, index=[0])

if st.button("See Input Data"):
   st.write(features)
   
if st.button("prediction"):
   result= predict_churn(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary)
   st.success('The Output is {}'.format(result))
   
if st.button("Prediction Probability"): 
   result= predict_churn_proba(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary)
   st.success('The Probability is {}'.format(result))
   s = "Exited" 
   if (result[0, 0] > result[0, 1]):
      s = "Not Exited"
   st.set_option('deprecation.showPyplotGlobalUse', False)
   d_nc = result[0, 0] * 360
   d_c = result[0, 1] * 360 
   c1 = (226, 33, 7)
   c2 = (20,20,80)
   make_pie([d_nc, d_c], s, [c2, c1], ['Probability(Not Exited): \n{0:.2f}%'.format(result[0, 0] * 100),
                                    'Probability (Exited): \n{0:.2f}%'.format(result[0, 1] * 100)])
   st.pyplot()
      
if st.button("About"):
   st.write('**This program tells us the probability of a customer staying Or leaving according to the input data.The Catboost Classifier does the classification work.**')
   st.write('**You can see with the prediction button that your customer belongs to class 1 or class 0 according to the input data.Class 0 = Not Exiced Class1 = Exicted.**')
   st.write('**You will also see the possibility of belonging to each class with the prediction Probability button.if  prediction Probability[0] > prediction Probability[1] customer Class=0**')
   st.write('This App Created by Navid Atashfaraz')
   st.image('Navid-a.jpg', width = 130) 




