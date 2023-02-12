import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('./model/pipe.pkl', 'rb'))
df = pickle.load(open('./data/df.pkl', 'rb'))

st.header('Car Price Predictor')

cols = df.keys().tolist()
cols.remove('selling_price')

input_values = [None] * len(cols)

for i in range(0, len(cols)):
    if df.dtypes[cols[i]] == 'O':
        input_values[i] = st.selectbox('{}'.format(
            cols[i]), df[cols[i]].value_counts().index.tolist())
    else:
        input_values[i] = st.number_input('{}'.format(cols[i]))


if st.button('Predict Price'):
    input = np.array([input_values])
    input = pd.DataFrame(input, columns=cols)

    y_pred = pipe.predict(input)
    st.title("Rs: {} Lakh".format(str(np.round(y_pred[0]/100000, 2))))
