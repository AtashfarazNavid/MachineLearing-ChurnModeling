import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time

st.write('Hello, *World!* :sunglasses:')
st.title('Customer Churn Prediction App')

audio_file = open('Soy.ogg', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

st.image('churn2.png', caption=None, width=None, use_column_width=False)
         

st.sidebar.header('User Input Parameters')

def user_input_features():
    CreditScore = st.sidebar.slider('Credit Score',350.00 ,650.52, 850.00 )
    Age = st.sidebar.slider('Age',18 ,38,92)
    Balance = st.sidebar.slider('Balance',0,250898)
    EstimatedSalary = st.sidebar.slider('Estimated Salary',12.00,100090.23,199992.48)
    HasCrCard = st.sidebar.radio("HasCrCard",[0,1])
    IsActiveMember = st.sidebar.radio("Active Member",[0,1])
    
    data = {'CreditScore': CreditScore,
            'Age': Age,
            'Balance': Balance,
            'HasCrCard':HasCrCard,
            'IsActiveMember':IsActiveMember,
            'EstimatedSalary':EstimatedSalary}        
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('Input parameters')
st.write(df)


#load data
NavidData = pd.read_csv("Churn.csv")
#Drop the unwanted columns and encoding variable
NavidData.drop(['RowNumber','CustomerId','Surname'], axis=1 ,inplace = True)

# Spliting data
X=NavidData[['CreditScore','Age','Balance','HasCrCard','IsActiveMember','EstimatedSalary']].values
Y=NavidData['Exited'].values 

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Not Exited or Exited ( NO = 0 , Yes = 1)')
st.subheader('if prediction Probability[0] > prediction Probability[1] customer Class=NO')

s = "Exited" 
if (prediction_proba[0, 0] > prediction_proba[0, 1]):
    s = "Not Exited"
    
st.set_option('deprecation.showPyplotGlobalUse', False)
d_nc = prediction_proba[0, 0] * 360
d_c = prediction_proba[0, 1] * 360

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
c1 = (226, 33, 7)
c2 = (20,20,80)

make_pie([d_nc, d_c], s, [c2, c1], ['Probability(Not Exited): \n{0:.2f}%'.format(prediction_proba[0, 0] * 100),
                                    'Probability (Exited): \n{0:.2f}%'.format(prediction_proba[0, 1] * 100)])

st.pyplot()


