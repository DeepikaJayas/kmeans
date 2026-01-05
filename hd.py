import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🛍️ Customer Segmentation using K-Means")
st.write("Mall Customers Segmentation based on **Income & Spending Score**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    kmeans_model = joblib.load("kmeans_model.pkl")
    scaler_model = joblib.load("scaler.pkl")
    return kmeans_model, scaler_model

kmeans, scaler = load_models()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# ---------------- SHOW DATA ----------------
if st.checkbox("📂 Show Dataset"):
    st.dataframe(df.head())

# ---------------- USER INPUT ----------------
st.subheader("🔢 Enter Customer Details")

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    max_value=300.0,
    step=1.0
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

# ---------------- PREDICTION ----------------
cluster = None

if st.button("🔍 Predict Customer Segment"):
    user_data = [[income, spending]]
    user_scaled = scaler.transform(user_data)
    cluster = int(kmeans.predict(user_scaled)[0])

    st.success(f"🧩 Customer belongs to **Cluster {cluster}**")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Customer Segments Visualization")

# Predict clusters for dataset
df_features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
df['Cluster'] = kmeans.predict(scaler.transform(df_features))

fig, ax = plt.subplots(figsize=(7, 5))

for c in sorted(df['Cluster'].unique()):
    ax.scatter(
        df[df['Cluster'] == c]['Annual Income (k$)'],
        df[df['Cluster'] == c]['Spending Score (1-100)'],
        label=f'Cluster {c}',
        alpha=0.7
    )

# Plot user point ONLY if predicted
if cluster is not None:
    ax.scatter(
        income,
        spending,
        marker='X',
        s=150,
        label='You'
    )

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation using K-Means")
ax.legend()

st.pyplot(fig)
