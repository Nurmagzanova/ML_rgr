from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 
import pickle
from sklearn import tree
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss 
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
from abc import ABC,abstractmethod
from typing import Tuple

class CART:
    """
    Реализация алгоритма дерева решений (CART).

    Параметры:
    - task_type (str): Тип задачи, 'classification' для классификации, 'regression' для регрессии.
    - max_depth (int): Максимальная глубина дерева. Если None, дерево строится до исчерпания данных.
    - min_samples_split (int): Минимальное количество образцов, необходимых для разделения внутреннего узла.

    Методы:
    - fit(X, y): Обучает модель на обучающих данных X и метках y.
    - predict(X): Прогнозирует метки для новых данных X.

    """

    def __init__(self, task_type='classification', max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task_type = task_type
        if task_type == 'classification':
            self.impurity_func = lambda x, y: self._gini(x, y)
        elif task_type == 'regression':
            self.impurity_func = lambda x, y: self._mse(x, y)
        else:
            raise ValueError("Неправильный выбор операции!")

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or len(X) <= self.min_samples_split:
            #если количество образцов меньше или равно минимальному количеству образцов для разбиения
            leaf_value = np.mean(y) if self.task_type == 'regression' else np.argmax(np.bincount(y)) 
            # возвращает массив с кол-вом каждого числа в массиве начиная с 0
            return {'leaf': True, 'value': leaf_value}

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None or best_threshold is None:
            leaf_value = np.mean(y) if self.task_type == 'regression' else np.argmax(np.bincount(y))
            return {'leaf': True, 'value': leaf_value}

        left_child_indices = X[:, best_feature] <= best_threshold
        right_child_indices = X[:, best_feature] > best_threshold

        node = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_child_indices], y[left_child_indices], depth + 1),
            'right': self._build_tree(X[right_child_indices], y[right_child_indices], depth + 1)
        }

        return node

    def _find_best_split(self, X, y):
        best_impurity = float('inf') if self.task_type == 'regression' else 1.0
        best_feature = None
        best_threshold = None #порог

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                impurity = self.impurity_func(y[left_indices], y[right_indices])
                #вычисляет нечистоту разбиения для текущего порога

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, left_y, right_y):
        left_gini = 1 - np.sum((np.bincount(left_y) / len(left_y))**2)
        right_gini = 1 - np.sum((np.bincount(right_y) / len(right_y))**2)
        gini = (len(left_y) * left_gini + len(right_y) * right_gini) / (len(left_y) + len(right_y))
        return gini

    #среднекв ошибка
    def _mse(self, left_y, right_y):
        left_mse = np.mean((left_y - np.mean(left_y))**2)
        right_mse = np.mean((right_y - np.mean(right_y))**2)
        mse = (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))
        return mse
#предсказывает набор х используя обученное дерево
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
#предсказ значение на основе текущего узла 
    def _traverse_tree(self, x, node):
        if 'leaf' in node:
            return node['value']


        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])


data = st.file_uploader("Выберите файл датасета", type=["csv"])
if data is not None:
    st.header("Датасет")
    df = pd.read_csv(data)
    st.dataframe(df)

    st.write("---")

    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", ['treeClassification'])

    st.title("Выбор признака")
    feature = st.selectbox("Выберите признак", df.columns)

    button_clicked = st.button("Обработка данных и предсказание")
    if button_clicked:
        st.header("Обработка данных")

        df.loc[(df['Fire Alarm'] == "Yes"), 'Fire Alarm'] = 1
        df.loc[(df['Fire Alarm'] == "No"), 'Fire Alarm'] = 0
        df['Fire Alarm'] = df['Fire Alarm'].astype(int)
            
        df[feature] = df[feature].astype(int)
        df = df.drop_duplicates()
        
        for i in df.columns[:-1]:
            df[i]=df[i].map(lambda x: np.random.uniform(int(df[i].min()), int(df[i].max())) if pd.isna(x) else x)

        outlier = df[df.columns[:-1]]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3-Q1
        data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(data_filtered.index.values)
        data_filtered = df[df.index.isin(index_list)]

        st.write("Очистка от выбросов: выполнено")

        scaler = StandardScaler()
        data_scaler = scaler.fit_transform(df.drop([feature], axis=1))
        st.write("Масштабирование: выполнено")

        y = df[feature]
        X = df.drop([feature], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

        nm = NearMiss()
        X, y = nm.fit_resample(X, y.ravel())

        st.write("Балансировка целевого признака: выполнено")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        st.success("Обработка завершена")

        st.header("Предсказание")

        
        classificator = CART(task_type='classification', max_depth=4, min_samples_split=5)
        classificator.fit(X_train.values, y_train)
        y_pred = classificator.predict(X_test.values)
        st.write(accuracy_score(y_test, y_pred))
        st.write('f1 = {}'.format(f1_score(y_test, y_pred)))
