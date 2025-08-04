import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("🚚 Delivery Delay Analysis Dashboard")

df = pd.read_csv('abc_delivery_data.csv')
st.write("### 📄 Dữ liệu gốc", df.head())

st.subheader("Biểu đồ 1: Giao hàng trễ theo khu vực")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='region', hue='delayed', ax=ax1)
plt.title('Late Delivery Rate by Region')
st.pyplot(fig1)

st.subheader("Biểu đồ 2: Thời tiết và thời gian giao hàng")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x='weather_condition', y='delivery_duration', ax=ax2)
plt.title('Weather affects delivery times')
st.pyplot(fig2)

st.subheader("Biểu đồ 3: Phân phối thời gian giao hàng")
fig3, ax3 = plt.subplots()
sns.histplot(df['delivery_duration'], bins=30, kde=True, ax=ax3)
plt.title('Delivery time distribution')
st.pyplot(fig3)

df.dropna(inplace=True)
df['delayed'] = df['delayed'].astype(int)

carrier_delayed_rate = df.groupby('carrier')['delayed'].mean().reset_index()

df = pd.get_dummies(df, columns=['weather_condition', 'region', 'carrier'], drop_first=True)

st.subheader("Biểu đồ 4: Ma trận tương quan giữa các biến")
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
plt.title('Correlation matrix between variables')
st.pyplot(fig4)

st.subheader("Biểu đồ 5: Tỉ lệ giao hàng trễ theo hãng vận chuyển")
fig5, ax5 = plt.subplots()
sns.barplot(data=carrier_delayed_rate, x='carrier', y='delayed', ax=ax5)
plt.title('Delay rate by shipping unit')
st.pyplot(fig5)

st.subheader("🌳 Mô hình dự đoán giao hàng trễ")
X = df.drop(columns=['delayed'])
y = df['delayed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Báo cáo phân loại:")
st.text(classification_report(y_test, y_pred))

st.subheader("Biểu đồ 6: Cây quyết định")
fig6, ax6 = plt.subplots(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['On time', 'Delayed'], ax=ax6)
plt.title('Decision tree for predicting late orders')
st.pyplot(fig6)