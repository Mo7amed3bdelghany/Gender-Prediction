import streamlit as st
import numpy as np
import pandas as pd
import joblib
# import base64
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

APP_TITLE = "🔍 Face Analysis & Gender Prediction Suite"
APP_SUBHEADER = "An Interactive Dashboard for Exploring Facial Features and Predicting Gender with Machine Learning"
LOCAL_IMAGE_PATH = Path("background.jpg")
MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("gender_classification_v7.csv")  
FEATURES = [
    "long_hair",
    "forehead_width_cm",
    "forehead_height_cm",
    "nose_wide",
    "nose_long",
    "lips_thin",
    "distance_nose_to_lip_long"
]
LABEL_MAP = {0: "Female", 1: "Male"}

# --------------- UTILS --------------- #


def load_model(model_path: Path):
    if not model_path.exists():
        st.error(f"Model file '{model_path}' not found.")
        st.stop()
    return joblib.load(model_path)

# @st.cache_data(show_spinner=False)
# def get_main_bg_css_from_local(img_path: Path) -> str:
#     if not img_path.exists():
#         return ""
#     img_bytes = img_path.read_bytes()
#     b64_encoded = base64.b64encode(img_bytes).decode()
#     return f"""
#     <style>
#     .stApp {{
#         background: url("data:image/jpg;base64,{b64_encoded}") no-repeat center center fixed;
#         background-size: cover;
#     }}
#     </style>
#     """


def predict(model, input_dict):
    X = np.array([[input_dict[feat] for feat in FEATURES]])
    y_pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][y_pred] if hasattr(model, "predict_proba") else None
    return y_pred, prob


# --------------- MAIN APP --------------- #
def main():
    st.set_page_config(APP_TITLE, "💡", layout="wide")
    # st.markdown(get_main_bg_css_from_local(LOCAL_IMAGE_PATH), unsafe_allow_html=True)

    st.title(APP_TITLE)
    st.markdown(f"### {APP_SUBHEADER}")
    st.markdown("---")

    menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Dashboard", "📈 Insights", "🧠 Predict Gender"])
    model = load_model(MODEL_PATH)

    # Optional: Load Data for reports
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame(columns=FEATURES + ['gender'])

    if menu == "🏠 Home":
        # st.image("photo_2025-07-15_01-30-37.jpg", use_column_width=True)
        st.markdown("""####
        Welcome to the **Face & Gender AI App.

        🌟 This tool lets you:
        - Predict gender based on facial traits.
        - Analyze trends with real data.
        - Visualize feature patterns by gender.
        - Interact with customizable reports.
        """)

    elif menu == "📊 Dashboard":
        st.header("📊 Data Overview")
        if df.empty:
            st.warning("No dataset available. Upload `data.csv` to enable dashboard features.")
        else:
            n_rows = st.slider('Choose Numbers of rows to show', min_value=5, max_value=len(df), step=1)
            columns_to_show = st.multiselect('Select Columns To Show', df.columns.tolist(), default=df.columns.tolist())
            st.write(df.iloc[:n_rows][columns_to_show])

            tabs = st.tabs(['Scatter plot', 'Histogram'])
            with tabs[0]:
                columns = st.columns(3)
                with columns[0]:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    x_columns = st.selectbox('Select column on X axis:', num_cols)
                with columns[1]:
                    y_columns = st.selectbox('Select column on Y axis:', num_cols)
                with columns[2]:
                    color = st.selectbox('Select column to be color', df.columns)
                fig_scatter = px.scatter(df, x=x_columns, y=y_columns, color=color)
                st.plotly_chart(fig_scatter)
            
            with tabs[1]:
                feature_to_plot = st.selectbox("Choose feature to plot", FEATURES)
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=feature_to_plot, hue='gender', kde=True, ax=ax)
                st.pyplot(fig)

    elif menu == "📈 Insights":
        st.header("🔍 Data Insights")
        if df.empty:
            st.warning("Please upload a dataset to generate insights.")
        else:
            st.markdown("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[FEATURES].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.markdown("#### Average Feature Values by Gender")
            st.dataframe(df.groupby("gender")[FEATURES].mean())

    elif menu == "🧠 Predict Gender":
        st.header("🎯 Gender Prediction")
        cols = st.columns(3)
        user_input = {
            "long_hair": cols[0].selectbox("Long Hair?", [0, 1]),
            "forehead_width_cm": cols[1].number_input("Forehead Width (cm)", 10.0, 25.0, 14.0, step=0.1),
            "forehead_height_cm": cols[2].number_input("Forehead Height (cm)", 3.0, 10.0, 5.4, step=0.1),
        }
        cols2 = st.columns(3)
        user_input["nose_wide"] = cols2[0].selectbox("Wide Nose?", [0, 1])
        user_input["nose_long"] = cols2[1].selectbox("Long Nose?", [0, 1])
        user_input["lips_thin"] = cols2[2].selectbox("Thin Lips?", [0, 1])
        user_input["distance_nose_to_lip_long"] = st.selectbox("Long Nose-to-Lip Distance?", [0, 1])

        if st.button("🔮 Predict Now"):
            label_idx, confidence = predict(model, user_input)
            gender = LABEL_MAP.get(label_idx, str(label_idx))
            st.success(f"**Predicted Gender:** {gender}")
            if confidence is not None:
                st.info(f"Model Confidence: {confidence * 100:.2f}%")
            st.dataframe(pd.DataFrame([user_input]))
            st.balloons() if gender == "Female" else st.snow()

    # Footer
    st.markdown("""<div style='text-align:center; padding-top:40px; font-size:0.8rem;'>
                 © 2025 • Built by <a href='https://github.com/Mo7amed3bdelghany' style='color:inherit;'>Mohamed Abdelghany</a></div>""",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
