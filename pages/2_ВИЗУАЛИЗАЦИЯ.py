import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("DataSet4_fil.xlsx")

st.title('Визуализация датасета')

st.header('Датасет для классификации - "Срабатывания датчика дыма"')

st.markdown('---')

st.write("Диаграмма с областями для температуры и влажности")

chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["Temperature[C]", "Humidity[%]"])
st.area_chart(chart_data)

st.write("Диаграмма рассеиния для уровеня H2,CO2 и этанола")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Raw H2", "Raw Ethanol", "eCO2[ppm]"])
st.scatter_chart(chart_data)

st.write("Гистограмма для численной концентрации твердых частиц различного диаметра")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["NC0.5", "NC1.0", "NC2.5"])
st.bar_chart(chart_data)

st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()
ax.hist(data['Fire Alarm'], bins=20)

st.pyplot(fig)