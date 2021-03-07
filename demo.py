import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
print("hello")
st.write('''#Diabetes prediction
Detect if some one has diabetes using machine learning''')
image=Image.open('C:/Users/Public/Pictures/Sample Pictures/Koala.jpg')
st.image(image,caption='ML', use_column_width=True)
df=pd.read_csv("C:/Users/SATISH KUMAR/Desktop/pdfjournals/datasets/diabetes1.csv")
st.subheader('DataInformation:')
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)
X=df.iloc[:,0:8].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
def get_user_input():
    pregnancies=st.sidebar.slider('pregnancies',0,17,3)
    glucose=st.sidebar.slider('glucose',0,199,117)
    bloodpressure=st.sidebar.slider('bloodpressure',0,122,72)
    skinthickness=st.sidebar.slider('skinthickness',0,99,23)
    insulin=st.sidebar.slider('insulin',0.0,846.0,30.5)
    bmi=st.sidebar.slider('bmi',0.0,67.1,32.0)
    diabetespedigreefunction=st.sidebar.slider('diabetespedigreefunction',0.078,2.42,0.3725)
    age=st.sidebar.slider('age',21,81,29)
    user_data={'pregnancies':pregnancies,
               'glucose':glucose,
               'bloodpressure':bloodpressure,
               'skinthickness':skinthickness,
               'insulin':insulin,
               'bmi':bmi,
               'diabetespedigreefunction':diabetespedigreefunction,
               'age':age}
    features=pd.DataFrame(user_data,index=[0])
    return features
user_input=get_user_input()
st.subheader('User Input:')
st.write(user_input)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
st.subheader('modeltestaccuracyscore:')
st.write(str(accuracy_score(y_test,rfc.predict(X_test))*100)+'%')
prediction=rfc.predict(user_input)
st.subheader('classification:')
st.write(prediction)








    

