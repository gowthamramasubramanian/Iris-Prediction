import pandas as pd
import numpy as np
from  sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import streamlit as st

X,y = load_iris(return_X_y=True, as_frame=True)
# print(X)
X_scaled = MinMaxScaler().fit_transform(X)
col = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X_scaled = pd.DataFrame(X_scaled, columns=col)
# print(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f" Accuracy is {accuracy_score(y_test, y_pred)}")

st.title("Predict Species")
sepal_len = st.text_input("Enter sepal length")
sepal_w = st.text_input("Enter sepal width")
petal_len = st.text_input("Enter petal length")
petal_w = st.text_input("Enter petal width")
print(sepal_len, sepal_w, petal_len, petal_w)

# df1 = pd.DataFrame({})
# df1['sepal length (cm)'] = sepal_len
# df1['sepal width (cm)'] = sepal_w
# df1['petal length (cm)'] = petal_len
# df1['petal width (cm)'] = petal_w

df1 = pd.DataFrame({'sepal length (cm)':float(sepal_len),
                   'sepal width (cm)':float(sepal_w),
                   'petal length (cm)':float(petal_len),
                   'petal width (cm)':float(petal_w)}, index=[0])
print(df1)
st.write(f" **Species belong to class** {model.predict(df1)}")





