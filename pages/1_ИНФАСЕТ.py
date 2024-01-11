import streamlit as st
import pandas as pd
import numpy as np

data2= pd.read_csv("DataSet4_fil.xlsx")
df2 = pd.DataFrame(data2)

st.title('Информация о датасетe')

st.header('Датасет для классификации - "Срабатывания датчика дыма"')
st.markdown('---')
st.dataframe(df2)

st.subheader('UTC')
st.markdown('Серия')

st.subheader('Temperature[C]')
st.markdown('Температура')

st.subheader('Humidity[%]')
st.markdown('Влажность')

st.subheader('TVOC[ppb]')
st.markdown('Индекс наличия летучих органических соединений')

st.subheader('eCO2[ppm]')
st.markdown('Количество CO2')

st.subheader('Raw H2')
st.markdown('Уровень H2')

st.subheader('Raw Ethanol')
st.markdown('Уровень этанола')

st.subheader('Pressure[hPa]')
st.markdown('Давление')

st.subheader('PM1.0')
st.markdown('Наличие частиц размером 1.0 микрометров и меньше')

st.subheader('PM2.5')
st.markdown('Наличие частиц размером 2.5 микрометров и меньше')

st.subheader('NC0.5')
st.markdown('Численная концентрация твердых частиц меньше 0.5 микрометров')

st.subheader('NC1.0')
st.markdown('Численная концентрация твердых частиц меньше 1.0 микрометров')

st.subheader('NC2.5')
st.markdown('Численная концентрация твердых частиц меньше 2.5 микрометров')

st.subheader('CNT')
st.markdown('Счетчик образцов')

st.subheader('Fire Alarm')
st.markdown('Если есть пожар, то 1, иначе 0')