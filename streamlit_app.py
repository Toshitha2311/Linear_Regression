import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from streamlit_option_menu import option_menu

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.hero-section {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 45px;
    border-radius: 25px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

.main-title {
    font-size: 45px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #f3f4f6;
}

.metric-card {
    background: #111827;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
}

.metric-card h2 {
    font-size: 35px;
}

.metric-card p {
    color: #d1d5db;
}

.prediction-box {
    background: linear-gradient(135deg,#16a34a,#22c55e);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin-top: 20px;
}

.stButton>button {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

.stDownloadButton>button {
    background: #16a34a;
    color: white;
    border-radius: 10px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DEFAULT DATASET
# ---------------------------------------------------

@st.cache_data
def load_default_data():

    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

    return pd.read_csv(url)

# ---------------------------------------------------
# TOP NAVBAR
# ---------------------------------------------------

selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Dataset",
        "Prediction",
        "Visualization",
        "About"
    ],
    icons=[
        "house",
        "table",
        "cpu",
        "bar-chart",
        "info-circle"
    ],
    orientation="horizontal",
    default_index=0
)

page = selected

# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------

st.sidebar.header("⚙️ Settings")

# Dataset options
dataset_option = st.sidebar.radio(
    "Choose Dataset",
    [
        "Default Boston Housing",
        "Upload Custom Dataset"
    ]
)

# Default dataset
if dataset_option == "Default Boston Housing":

    df = load_default_data()

    st.sidebar.success(
        "Default Boston Housing dataset loaded"
    )

# Custom dataset
else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.sidebar.success(
            "Custom dataset uploaded"
        )

    else:

        st.warning(
            "Please upload a CSV file"
        )

        st.stop()

# ---------------------------------------------------
# OUTLIER SETTINGS
# ---------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Outlier Settings")

show_outliers = st.sidebar.checkbox(
    "Show Outlier Analysis",
    value=True
)

remove_outliers = st.sidebar.checkbox(
    "Remove Outliers (IQR)",
    value=False
)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

outlier_column = st.sidebar.selectbox(
    "Outlier Column",
    numeric_columns
)

# ---------------------------------------------------
# REMOVE OUTLIERS FUNCTION
# ---------------------------------------------------

def remove_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)

    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    filtered_data = data[
        (data[column] >= lower) &
        (data[column] <= upper)
    ]

    return filtered_data

# Apply outlier removal
if remove_outliers:

    old_rows = df.shape[0]

    df = remove_outliers_iqr(
        df,
        outlier_column
    )

    new_rows = df.shape[0]

    removed_rows = old_rows - new_rows

    st.sidebar.success(
        f"{removed_rows} outliers removed"
    )

# ---------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model Settings")

target_col = st.sidebar.selectbox(
    "Target Variable",
    numeric_columns,
    index=len(numeric_columns)-1
)

feature_columns = [
    col for col in numeric_columns
    if col != target_col
]

selected_features = st.sidebar.multiselect(
    "Select Features",
    feature_columns,
    default=feature_columns
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Linear Regression",
        "Random Forest",
        "Decision Tree",
        "Ridge Regression"
    ]
)

use_scaler = st.sidebar.checkbox(
    "Use StandardScaler",
    value=False
)

use_cv = st.sidebar.checkbox(
    "Use Cross Validation",
    value=False
)

cv_folds = st.sidebar.slider(
    "CV Folds",
    3,
    10,
    5
)

# ---------------------------------------------------
# TRAIN MODEL FUNCTION
# ---------------------------------------------------

def train_model():

    data = df.copy()

    imputer = SimpleImputer(strategy="mean")

    data[numeric_columns] = imputer.fit_transform(
        data[numeric_columns]
    )

    X = data[selected_features]

    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model selection
    if model_choice == "Linear Regression":

        model = LinearRegression()

    elif model_choice == "Random Forest":

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    elif model_choice == "Decision Tree":

        model = DecisionTreeRegressor(
            random_state=42
        )

    else:

        model = Ridge(alpha=1.0)

    steps = []

    if use_scaler:

        steps.append(
            ('scaler', StandardScaler())
        )

    steps.append(
        ('model', model)
    )

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    cv_score = None

    if use_cv:

        cv_score = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_folds,
            scoring='r2'
        )

    return (
        pipeline,
        X,
        y,
        X_test,
        y_test,
        y_pred,
        r2,
        rmse,
        mae,
        cv_score
    )

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

with st.spinner("Training Model..."):

    (
        pipeline,
        X,
        y,
        X_test,
        y_test,
        y_pred,
        r2,
        rmse,
        mae,
        cv_score
    ) = train_model()

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero-section">

    <div class="main-title">
    🏠 Housing Price Prediction
    </div>

    <div class="subtitle">
    Predict house prices using advanced Machine Learning models with beautiful visualizations.
    </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
        <h2>{df.shape[0]}</h2>
        <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
        <h2>{df.shape[1]}</h2>
        <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
        <h2>{r2:.3f}</h2>
        <p>R² Score</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("✨ Features")

    st.write("""
    ✅ Multiple ML Models  
    ✅ Upload Custom Dataset  
    ✅ Outlier Detection & Removal  
    ✅ Batch Predictions  
    ✅ Correlation Heatmaps  
    ✅ Feature Importance  
    ✅ Cross Validation  
    ✅ Interactive UI  
    """)

# ---------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------

elif page == "Dataset":

    st.title("📊 Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.dataframe(df)

    st.subheader("Dataset Statistics")

    st.write(df.describe())

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "⬇ Download Dataset",
        data=csv,
        file_name="dataset.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

elif page == "Prediction":

    st.title("🔮 Predict House Price")

    input_data = {}

    cols = st.columns(3)

    for idx, col_name in enumerate(selected_features):

        col = cols[idx % 3]

        input_data[col_name] = col.number_input(
            col_name,
            value=float(df[col_name].mean())
        )

    if st.button("Predict Price"):

        input_df = pd.DataFrame([input_data])

        prediction = pipeline.predict(input_df)[0]

        st.markdown(f"""
        <div class="prediction-box">
        Predicted Price<br>
        ${prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# VISUALIZATION PAGE
# ---------------------------------------------------

elif page == "Visualization":

    st.title("📈 Visualizations")

    plt.style.use("dark_background")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation",
        "Actual vs Predicted",
        "Feature Importance",
        "Outlier Analysis"
    ])

    # ---------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------

    with tab1:

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap",
            fontsize=16,
            fontweight='bold'
        )

        st.pyplot(fig)

    # ---------------------------------------------------
    # ACTUAL VS PREDICTED
    # ---------------------------------------------------

    with tab2:

        st.subheader("📌 Actual vs Predicted with Regression Line")

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        # Scatter Plot
        ax.scatter(
            y_test,
            y_pred,
            color="#3b82f6",
            alpha=0.7,
            s=70,
            label="Predicted Points"
        )

        # Regression Line
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)

        ax.plot(
            y_test,
            p(y_test),
            color="#ef4444",
            linewidth=3,
            label="Regression Line"
        )

        # Perfect Prediction Line
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="#22c55e",
            linestyle="--",
            linewidth=2,
            label="Perfect Prediction"
        )

        ax.set_xlabel(
            "Actual Values",
            fontsize=12,
            fontweight='bold'
        )

        ax.set_ylabel(
            "Predicted Values",
            fontsize=12,
            fontweight='bold'
        )

        ax.set_title(
            "Actual vs Predicted",
            fontsize=16,
            fontweight='bold'
        )

        ax.grid(alpha=0.3)

        ax.legend()

        st.pyplot(fig)

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------

    with tab3:

        try:

            model = pipeline.named_steps['model']

            if hasattr(model, "coef_"):

                importance = model.coef_

            elif hasattr(model, "feature_importances_"):

                importance = model.feature_importances_

            else:

                raise Exception

            coef_df = pd.DataFrame({

                "Feature": selected_features,
                "Importance": importance

            })

            coef_df = coef_df.sort_values(
                by="Importance",
                key=abs,
                ascending=False
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            bars = ax.barh(
                coef_df["Feature"],
                coef_df["Importance"]
            )

            for bar in bars:

                width = bar.get_width()

                ax.text(
                    width,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    va='center'
                )

            ax.set_title(
                "Feature Importance",
                fontsize=16,
                fontweight='bold'
            )

            st.pyplot(fig)

            st.dataframe(coef_df)

        except:

            st.info(
                "Feature importance not available for this model."
            )

    # ---------------------------------------------------
    # OUTLIER ANALYSIS
    # ---------------------------------------------------

    with tab4:

        if show_outliers:

            st.subheader(
                "📌 Outlier Analysis"
            )

            col1, col2 = st.columns(2)

            # Boxplot
            with col1:

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                sns.boxplot(
                    y=df[outlier_column],
                    color="skyblue",
                    ax=ax
                )

                ax.set_title(
                    f"Boxplot - {outlier_column}"
                )

                st.pyplot(fig)

            # Distribution plot
            with col2:

                fig2, ax2 = plt.subplots(
                    figsize=(7, 5)
                )

                sns.histplot(
                    df[outlier_column],
                    kde=True,
                    color="orange",
                    ax=ax2
                )

                ax2.set_title(
                    f"Distribution - {outlier_column}"
                )

                st.pyplot(fig2)

# ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------

elif page == "About":

    st.title("ℹ️ About")

    st.write("""
    ## Housing Price Prediction App

    This project predicts housing prices using
    Machine Learning models.

    ### Technologies Used

    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - Matplotlib
    - Seaborn

    ### Models Included

    - Linear Regression
    - Random Forest
    - Decision Tree
    - Ridge Regression
    """)

    st.subheader("📌 Model Metrics")

    st.write(f"R² Score: {r2:.3f}")

    st.write(f"RMSE: {rmse:.3f}")

    st.write(f"MAE: {mae:.3f}")

    if cv_score is not None:

        st.write(
            f"Cross Validation Mean Score: {cv_score.mean():.4f}"
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
<center>
🏠 Housing Price Predictor • Built with Streamlit
</center>
""", unsafe_allow_html=True)