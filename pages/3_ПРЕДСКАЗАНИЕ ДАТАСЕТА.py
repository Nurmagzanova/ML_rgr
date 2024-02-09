import pandas as pd 
import numpy as np 
import pickle
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss 
import io

def Prediction(model, X_test):

    y_pred = model.predict(X_test)
    y_pred=np.asarray(y_pred)
    with io.BytesIO() as buffer:
        np.savetxt(buffer, y_pred, delimiter=",")
        st.download_button(label = 'Скачать предсказания',
                            data = buffer,
                            file_name = 'predictions.csv',
                            mime = 'text/csv')



data = st.file_uploader("Выберите файл датасета", type=["csv","xls"])
if data is not None:
    st.header("Датасет")
    df = pd.read_csv(data)
    st.dataframe(df)

    st.write("---")

    feature = st.selectbox("Выберите предсказываемый признак",df.columns)

    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", [ 'Knn', 'Kmeans','Boosting', 'Bagging','Stacking', 'MLP' ])

    button_clicked = st.button("Обработка данных и предсказание")
    if button_clicked:
        st.header("Обработка данных")

        df = df.drop_duplicates()

        scaler = StandardScaler()
        data_scaler = scaler.fit_transform(df.drop(feature, axis=1))

        y = df[feature]
        X = df.drop([feature], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

        nm = NearMiss()
        X, y = nm.fit_resample(X, y.ravel())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        st.success("Обработка завершена")

        st.header("Предсказание")

        if model_type is not None:
            if model_type == "Выберите модель":
                st.write('Для получения предсказания')
            elif model_type == "Knn":
                    with open('models/knn.pkl', 'rb') as file:
                        knn_model = pickle.load(file)
                    Prediction(knn_model,X_test)
            elif model_type == "Kmeans":
                    with open('models/kmeans.pkl', 'rb') as file:
                        kmeans_model = pickle.load(file)
                    Prediction(kmeans_model,X_test)
            elif model_type == "Boosting":
                    with open('models/boosting.pkl', 'rb') as file:
                        boos_model = pickle.load(file)
                    Prediction(boos_model,X_test)
            elif model_type == "Bagging":
                    with open('models/bagging.pkl', 'rb') as file:
                        bagg_model = pickle.load(file)
                    Prediction(bagg_model,X_test)
            elif model_type == "Stacking":
                    with open('models/stacking.pkl', 'rb') as file:
                        stac_model = pickle.load(file)
                    Prediction(stac_model,X_test)
            elif model_type == "MLP":
                    with open('models/mlp.pkl', 'rb') as file:
                        mlp_model = pickle.load(file)
                    Prediction(mlp_model,X_test)
            elif model_type == "Nearest Neighbors Regression":
                    with open('models/knn (1).pkl', 'rb') as file:
                        mlp_model = pickle.load(file)
                    Prediction(mlp_model,X_test)
            elif model_type == "Nearest Centroid Classifier":
                    with open('models/clf.pkl', 'rb') as file:
                        mlp_model = pickle.load(file)
                    Prediction(mlp_model,X_test)