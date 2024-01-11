import pandas as pd 
import numpy as np 
import math
import pickle
from sklearn.model_selection import train_test_split 
import streamlit as st 


df= pd.read_csv('DataSet4_fil.csv')

if df is not None:
    st.header("Датасет")
    st.dataframe(df)

    st.write("---")

    df = df.drop('Unnamed: 0',axis=1)

    st.title("Fire Alarm Prediction") 
    list=[]

    for i in df.columns[:-1]:
        a = st.slider(i,int(df[i].min()), int(math.ceil(df[i].max())),int(df[i].max()/2))
        list.append(a)

    list.append(a)
    list = np.array(list).reshape(1,-1)
    list=list.tolist()

    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", ['Knn', 'Kmeans', 'Boosting', 'Bagging','Stacking', 'MLP' ])

    button_clicked = st.button("Предсказать")
    if button_clicked:
        if model_type is not None:
            if model_type == "Knn":
                with open('models/knn.pkl', 'rb') as file:
                    knn_model = pickle.load(file)
                if knn_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "Kmeans":
                with open('models/kmeans.pkl', 'rb') as file:
                    kmeans_model = pickle.load(file)
                if kmeans_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "Boosting":
                with open('models/boosting.pkl', 'rb') as file:
                    boos_model = pickle.load(file)
                if boos_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "Bagging":
                with open('models/bagging.pkl', 'rb') as file:
                    bagg_model = pickle.load(file)
                if bagg_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "Stacking":
                with open('models/stacking.pkl', 'rb') as file:
                    stac_model = pickle.load(file)
                if stac_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "MLP":
                with open('models/mlp.pkl', 'rb') as file:
                    mlp_model = pickle.load(file)
                if mlp_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")