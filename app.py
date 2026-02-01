import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")

VALIDATION_RANGES = {
    "age": (18, 100, "years"),
    "alcohol_consumption_per_week": (0, 50, "units"),
    "physical_activity_minutes_per_week": (0, 1000, "minutes"),
    "diet_score": (0, 10, "score"),
    "sleep_hours_per_day": (0, 24, "hours"),
    "screen_time_hours_per_day": (0, 24, "hours"),
    "bmi": (10, 60, "kg/m²"),
    "waist_to_hip_ratio": (0.5, 2.0, "ratio"),
    "systolic_bp": (70, 250, "mmHg"),
    "diastolic_bp": (40, 150, "mmHg"),
    "heart_rate": (30, 200, "bpm"),
    "cholesterol_total": (100, 500, "mg/dL"),
    "hdl_cholesterol": (10, 150, "mg/dL"),
    "ldl_cholesterol": (30, 400, "mg/dL"),
    "triglycerides": (30, 1000, "mg/dL"),
}

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "ethnicity": ["Caucasian", "African American", "Asian", "Hispanic", "Other"],
    "education_level": ["No formal", "Highschool", "Graduate", "Postgraduate"],
    "income_level": ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    "smoking_status": ["Never", "Former", "Current"],
    "employment_status": ["Employed", "Unemployed", "Self-Employed", "Retired"],
}

FEATURE_TOOLTIPS = {
    "age": "Patient's age in years (18-100)",
    "bmi": "Body Mass Index = weight(kg) / height(m)²",
    "waist_to_hip_ratio": "Waist circumference / Hip circumference",
    "systolic_bp": "Upper blood pressure reading (normal: ~120)",
    "diastolic_bp": "Lower blood pressure reading (normal: ~80)",
    "cholesterol_total": "Total blood cholesterol (normal: <200 mg/dL)",
    "hdl_cholesterol": "Good cholesterol (higher is better, normal: >40)",
    "ldl_cholesterol": "Bad cholesterol (lower is better, normal: <100)",
    "triglycerides": "Blood fat levels (normal: <150 mg/dL)",
    "diet_score": "Diet quality score from 0 (poor) to 10 (excellent)",
    "physical_activity_minutes_per_week": "Total weekly exercise minutes (recommended: ≥150)",
    "sleep_hours_per_day": "Average daily sleep hours (recommended: 7-9)",
    "alcohol_consumption_per_week": "Weekly alcohol consumption in standard units",
    "screen_time_hours_per_day": "Average daily screen time in hours",
    "heart_rate": "Resting heart rate in beats per minute (normal: 60-100)",
}


@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
        lgbm_model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        catboost_model = joblib.load(MODELS_DIR / "catboost_model.pkl")

        # Try to load Random Forest (optional - large file)
        rf_model = None
        if (MODELS_DIR / "rf_model.pkl").exists():
            try:
                rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
            except Exception:
                pass  # Skip RF if loading fails

        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        ensemble_data = joblib.load(MODELS_DIR / "ensemble_weights.pkl")
        feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        ordinal_mappings = joblib.load(MODELS_DIR / "ordinal_mappings.pkl")
        nominal_features = joblib.load(MODELS_DIR / "nominal_features.pkl")

        # Build models list and adjust weights if RF is missing
        models = [xgb_model, lgbm_model, catboost_model]
        weights = ensemble_data["weights"][:3]  # First 3 weights

        if rf_model is not None:
            models.append(rf_model)
            weights = ensemble_data["weights"]  # All 4 weights
        else:
            # Normalize weights to sum to 1 when RF is excluded
            weights = np.array(weights)
            weights = weights / weights.sum()

        return {
            "models": models,
            "scaler": scaler,
            "weights": weights,
            "ensemble_auc": ensemble_data["ensemble_auc"],
            "feature_columns": feature_columns,
            "ordinal_mappings": ordinal_mappings,
            "nominal_features": nominal_features,
            "rf_included": rf_model is not None,
        }
    except FileNotFoundError as e:
        st.error(f"Models not found: {e}")
        return None


def create_advanced_features(df):
    df = df.copy()

    df["cholesterol_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)

    df["mean_arterial_pressure"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df["lifestyle_score"] = (
        df["diet_score"] * 0.3
        + (df["physical_activity_minutes_per_week"] / 150) * 0.3
        + (df["sleep_hours_per_day"] / 8) * 0.2
        + ((4 - df["alcohol_consumption_per_week"]) / 4) * 0.2
    )

    df["risk_factors_count"] = (
        (df["age"] >= 45).astype(int)
        + (df["bmi"] >= 30).astype(int)
        + (df["family_history_diabetes"] == 1).astype(int)
        + (df["hypertension_history"] == 1).astype(int)
        + (df["smoking_status"].isin(["Current", "Former"])).astype(int)
        + (df["physical_activity_minutes_per_week"] < 150).astype(int)
    )

    df["age_bmi_interaction"] = df["age"] * df["bmi"] / 100

    df["trig_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)

    df["metabolic_syndrome_score"] = (
        ((df["systolic_bp"] >= 130) | (df["diastolic_bp"] >= 85)).astype(int)
        + (df["triglycerides"] >= 150).astype(int)
        + (df["hdl_cholesterol"] < 40).astype(int)
        + (df["bmi"] >= 30).astype(int)
        + (df["family_history_diabetes"] == 1).astype(int)
    )

    df["glucose_risk_proxy"] = (
        df["age"] * 0.3 + df["bmi"] * 0.4 + df["systolic_bp"] / 100 * 0.3
    )

    df["cardiovascular_risk"] = (df["systolic_bp"] - 120) / 20 + (
        df["ldl_cholesterol"] - 100
    ) / 40

    df["age_squared"] = df["age"] ** 2

    df["activity_age_ratio"] = df["physical_activity_minutes_per_week"] / (
        df["age"] + 1
    )

    df["chol_bp_risk"] = (df["cholesterol_total"] / 200) * (df["systolic_bp"] / 120)

    df["protective_factors"] = (
        df["physical_activity_minutes_per_week"] / 150
        + df["diet_score"] / 10
        + (8 - abs(df["sleep_hours_per_day"] - 7.5)) / 8
    )

    df["diet_activity_synergy"] = df["diet_score"] * np.log(
        df["physical_activity_minutes_per_week"] + 1
    )

    return df


def preprocess_input(input_data, artifacts):
    """Preprocess user input through the complete pipeline."""
    df = pd.DataFrame([input_data])

    df_enhanced = create_advanced_features(df)

    for col, mapping in artifacts["ordinal_mappings"].items():
        df_enhanced[col] = df_enhanced[col].map(mapping)

    df_encoded = pd.get_dummies(
        df_enhanced, columns=artifacts["nominal_features"], drop_first=True
    )

    for col in artifacts["feature_columns"]:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[artifacts["feature_columns"]]

    df_scaled = pd.DataFrame(
        artifacts["scaler"].transform(df_encoded), columns=df_encoded.columns
    )

    return df_scaled


def calculate_ensemble_prediction(X, models, weights):
    predictions = []
    for model in models:
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba)

    # Weighted average of available models
    ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
    return ensemble_pred[0]


def validate_input(key, value):
    if key in VALIDATION_RANGES:
        min_val, max_val, unit = VALIDATION_RANGES[key]

        if not (min_val <= value <= max_val):
            return False, f"Value must be between {min_val} and {max_val} {unit}"
    return True, ""


def main():
    st.title("Diabetes Risk Prediction System")
    st.markdown(
        """
    This application uses uses Artificial Intelligence to predict diabetes risk based on health metrics and lifestyle factors.
    """
    )

    artifacts = load_models()
    if artifacts is None:
        st.stop()

    st.html(
        """
    XGBoost: <b style="color:green;">Live</b> <br>  
    LightGBM: <b style="color:green;">Live</b> <br>  
    CatBoost: <b style="color:green;">Live</b> <br>  
    Random Forest: """
        + (
            '<b style="color:green;">Live</b>'
            if artifacts.get("rf_included", False)
            else '<b style="color:orange;">Offline</b>'
        )
        + """
    """
    )

    st.header("Enter Patient Information")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demographics", "Lifestyle", "Medical History", "Lab Results"]
    )

    user_input = {}
    validation_errors = []

    with tab1:
        st.subheader("Demographic Information")
        col1, col2 = st.columns(2)

        with col1:
            user_input["age"] = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=45,
                help=FEATURE_TOOLTIPS["age"],
            )
            user_input["gender"] = st.selectbox(
                "Gender", options=CATEGORICAL_OPTIONS["gender"]
            )
            user_input["ethnicity"] = st.selectbox(
                "Ethnicity", options=CATEGORICAL_OPTIONS["ethnicity"]
            )

        with col2:
            user_input["education_level"] = st.selectbox(
                "Education Level", options=CATEGORICAL_OPTIONS["education_level"]
            )
            user_input["income_level"] = st.selectbox(
                "Income Level", options=CATEGORICAL_OPTIONS["income_level"]
            )
            user_input["employment_status"] = st.selectbox(
                "Employment Status", options=CATEGORICAL_OPTIONS["employment_status"]
            )

    with tab2:
        st.subheader("Lifestyle Factors")
        col1, col2 = st.columns(2)

        with col1:
            user_input["physical_activity_minutes_per_week"] = st.number_input(
                "Physical Activity (min/week)",
                min_value=0,
                max_value=1000,
                value=150,
                help=FEATURE_TOOLTIPS["physical_activity_minutes_per_week"],
            )
            user_input["diet_score"] = st.slider(
                "Diet Quality Score",
                min_value=0,
                max_value=10,
                value=5,
                help=FEATURE_TOOLTIPS["diet_score"],
            )
            user_input["sleep_hours_per_day"] = st.number_input(
                "Sleep (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=7.5,
                step=0.5,
                help=FEATURE_TOOLTIPS["sleep_hours_per_day"],
            )

        with col2:
            user_input["smoking_status"] = st.selectbox(
                "Smoking Status", options=CATEGORICAL_OPTIONS["smoking_status"]
            )
            user_input["alcohol_consumption_per_week"] = st.number_input(
                "Alcohol Consumption (units/week)",
                min_value=0,
                max_value=50,
                value=2,
                help=FEATURE_TOOLTIPS["alcohol_consumption_per_week"],
            )
            user_input["screen_time_hours_per_day"] = st.number_input(
                "Screen Time (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
            )

    with tab3:
        st.subheader("Medical History")
        col1, col2 = st.columns(2)

        with col1:
            user_input["family_history_diabetes"] = st.selectbox(
                "Family History of Diabetes",
                options=["No", "Yes"],
                format_func=lambda x: x,
            )
            user_input["family_history_diabetes"] = (
                1 if user_input["family_history_diabetes"] == "Yes" else 0
            )

            user_input["hypertension_history"] = st.selectbox(
                "History of Hypertension", options=["No", "Yes"]
            )
            user_input["hypertension_history"] = (
                1 if user_input["hypertension_history"] == "Yes" else 0
            )

        with col2:
            user_input["cardiovascular_history"] = st.selectbox(
                "Cardiovascular History", options=["No", "Yes"]
            )
            user_input["cardiovascular_history"] = (
                1 if user_input["cardiovascular_history"] == "Yes" else 0
            )

        st.markdown("**Body Measurements**")
        col1, col2, col3 = st.columns(3)
        with col1:
            height_cm = st.number_input(
                "Height (cm)",
                min_value=100.0,
                max_value=250.0,
                value=170.0,
                step=0.1,
                help="Patient's height in centimeters",
            )
        with col2:
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=30.0,
                max_value=250.0,
                value=70.0,
                step=0.1,
                help="Patient's weight in kilograms",
            )
        with col3:
            height_m = height_cm / 100
            calculated_bmi = weight_kg / (height_m**2)
            st.metric(
                "Calculated BMI",
                f"{calculated_bmi:.2f}",
                help="Body Mass Index = weight(kg) / height(m)²",
            )

        user_input["bmi"] = calculated_bmi

    with tab4:
        st.subheader("Laboratory Results")

        st.markdown("**Blood Pressure**")
        col1, col2, col3 = st.columns(3)
        with col1:
            user_input["systolic_bp"] = st.number_input(
                "Systolic BP",
                min_value=70,
                max_value=250,
                value=120,
                help=FEATURE_TOOLTIPS["systolic_bp"],
            )
        with col2:
            user_input["diastolic_bp"] = st.number_input(
                "Diastolic BP",
                min_value=40,
                max_value=150,
                value=80,
                help=FEATURE_TOOLTIPS["diastolic_bp"],
            )
        with col3:
            user_input["heart_rate"] = st.number_input(
                "Heart Rate (bpm)", min_value=30, max_value=200, value=72
            )

        st.markdown("**Physical Measurements**")
        col1, col2 = st.columns(2)
        with col1:
            user_input["waist_to_hip_ratio"] = st.number_input(
                "Waist-to-Hip Ratio",
                min_value=0.5,
                max_value=2.0,
                value=0.85,
                step=0.01,
                help=FEATURE_TOOLTIPS["waist_to_hip_ratio"],
            )

        st.markdown("**Cholesterol Panel**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            user_input["cholesterol_total"] = st.number_input(
                "Total Cholesterol",
                min_value=100,
                max_value=500,
                value=200,
                help=FEATURE_TOOLTIPS["cholesterol_total"],
            )
        with col2:
            user_input["hdl_cholesterol"] = st.number_input(
                "HDL (Good)",
                min_value=10,
                max_value=150,
                value=50,
                help=FEATURE_TOOLTIPS["hdl_cholesterol"],
            )
        with col3:
            user_input["ldl_cholesterol"] = st.number_input(
                "LDL (Bad)",
                min_value=30,
                max_value=400,
                value=100,
                help=FEATURE_TOOLTIPS["ldl_cholesterol"],
            )
        with col4:
            user_input["triglycerides"] = st.number_input(
                "Triglycerides",
                min_value=30,
                max_value=1000,
                value=150,
                help=FEATURE_TOOLTIPS["triglycerides"],
            )

    for key, value in user_input.items():
        is_valid, error_msg = validate_input(key, value)
        if not is_valid:
            validation_errors.append(f"**{key}**: {error_msg}")

    if validation_errors:
        st.error("Please correct the following input errors:")
        for error in validation_errors:
            st.write(f"- {error}")

    st.markdown("---")
    if st.button(
        "Predict Diabetes Risk", type="primary", disabled=len(validation_errors) > 0
    ):
        with st.spinner("Analyzing health data..."):
            try:
                X_processed = preprocess_input(user_input, artifacts)

                risk_probability = calculate_ensemble_prediction(
                    X_processed, artifacts["models"], artifacts["weights"]
                )

                if risk_probability < 0.3:
                    risk_level = "Low"
                    risk_color = "green"
                elif risk_probability < 0.6:
                    risk_level = "Moderate"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    risk_color = "red"

                st.markdown("### Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Probability", f"{risk_probability * 100:.2f}%")
                with col2:
                    st.metric("Risk Level", f"{risk_level}")
                with col3:
                    st.metric("Model Confidence", f"{artifacts['ensemble_auc']:.2%}")

                st.markdown("### Interpretation")
                if risk_level == "Low":
                    st.info(
                        """
                    **Low Risk**: Based on the provided health metrics, the model indicates a low risk 
                    of diabetes. Continue maintaining a healthy lifestyle with regular exercise, balanced 
                    diet, and routine health checkups.
                    """
                    )
                elif risk_level == "Moderate":
                    st.warning(
                        """
                    **Moderate Risk**: The model indicates a moderate risk of diabetes. Consider:
                    - Consulting with a healthcare provider for further evaluation
                    - Improving diet quality and increasing physical activity
                    - Regular monitoring of blood glucose levels
                    - Maintaining a healthy weight
                    """
                    )
                else:
                    st.error(
                        """
                    **High Risk**: The model indicates a high risk of diabetes. Immediate action recommended:
                    - Consult with a healthcare provider as soon as possible
                    - Request comprehensive diabetes screening tests
                    - Implement lifestyle changes under medical supervision
                    - Monitor health metrics closely
                    """
                    )

                st.markdown("---")
                st.caption(
                    """
                **Disclaimer**: This prediction is for informational purposes only and should not 
                replace professional medical advice, diagnosis, or treatment. Always consult with a 
                qualified healthcare provider for proper medical evaluation and care.
                """
                )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
