import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing artifacts
model = joblib.load("final_beverage_model.pkl")
expected_columns = joblib.load("expected_columns.pkl")
label_encoder_y = joblib.load("label_encoder_y.pkl")

# Manual label encodings used during training
label_encodings = {
    'age_group': {'18-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56-70': 4},
    'income_levels': {'<10L': 1, '10L - 15L': 2, '16L - 25L': 3, '26L - 35L': 4, '> 35L': 5, 'Not Reported': 0},
    'health_concerns': {
        'High (Very health-conscious)': 0,
        'Low (Not very concerned)': 1,
        'Medium (Moderately health-conscious)': 2
    },
    'consume_frequency(weekly)': {'0-2 times': 1, '3-4 times': 2, '5-7 times': 3},
    'preferable_consumption_size': {'Small (250 ml)': 0, 'Large (1L)': 1, 'Medium (500 ml)': 2}
}

# Fixed hidden scores
bsi_score = 50.0
cf_ab_score = 50.0
zas_score = 50.0

# Feature options
feature_options = {
    'age_group': ['18-25', '26-35', '36-45', '46-55', '56-70'],
    'gender': ['F', 'M'],
    'zone': ['Metro', 'Rural', 'Semi-Urban', 'Urban'],
    'occupation': ['Entrepreneur', 'Retired', 'Student', 'Working Professional'],
    'income_levels': ['<10L', '10L - 15L', '16L - 25L', '26L - 35L', '> 35L', 'Not Reported'],
    'health_concerns': ['High (Very health-conscious)', 'Medium (Moderately health-conscious)', 'Low (Not very concerned)'],
    'consume_frequency(weekly)': ['0-2 times', '3-4 times', '5-7 times'],
    'preferable_consumption_size': ['Small (250 ml)', 'Medium (500 ml)', 'Large (1L)'],
    'flavor_preference': ['Exotic', 'Traditional'],
    'typical_consumption_situations': ['Active', 'Casual', 'Social'],
    'current_brand': ['Established', 'Newcomer'],
    'purchase_channel': ['Online', 'Retail Store'],
    'packaging_preference': ['Eco-Friendly', 'Premium', 'Simple'],
    'awareness_of_other_brands': ['0 to 1', '2 to 3', '4+'],
    'reasons_for_choosing_brands': ['Availability', 'Brand Reputation', 'Price', 'Quality']
}

# Page configuration and styling
st.set_page_config(page_title="Beverage Price Predictor", layout="wide")

# Custom CSS for beautiful background and styling
st.markdown("""
    <style>
        body {
            background-color: #f2f6fa;
        }
        .main {
            background-color: #f2f6fa;
            padding: 2rem;
        }
        h1 {
            color: #2f4fdb;
        }
        .stButton>button {
            background-color: #2f4fdb;
            color: white;
            border-radius: 5px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🧃 CodeX Beverage: Price Prediction</h1>", unsafe_allow_html=True)

# Form
with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)

    # Row 1
    with col1:
        age_group = st.selectbox("Age", feature_options['age_group'])
    with col2:
        gender = st.selectbox("Gender", feature_options['gender'])
    with col3:
        zone = st.selectbox("Zone", feature_options['zone'])
    with col4:
        occupation = st.selectbox("Occupation", feature_options['occupation'])

    # Row 2
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        income = st.selectbox("Income Level (In L)", feature_options['income_levels'])
    with col6:
        freq = st.selectbox("Consume Frequency(weekly)", feature_options['consume_frequency(weekly)'])
    with col7:
        brand = st.selectbox("Current Brand", feature_options['current_brand'])
    with col8:
        size = st.selectbox("Preferable Consumption Size", feature_options['preferable_consumption_size'])

    # Row 3
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        awareness = st.selectbox("Awareness of other brands", feature_options['awareness_of_other_brands'])
    with col10:
        reason = st.selectbox("Reasons for choosing brands", feature_options['reasons_for_choosing_brands'])
    with col11:
        flavor = st.selectbox("Flavor Preference", feature_options['flavor_preference'])
    with col12:
        channel = st.selectbox("Purchase Channel", feature_options['purchase_channel'])

    # Row 4
    col13, col14, col15 = st.columns(3)
    with col13:
        packaging = st.selectbox("Packaging Preference", feature_options['packaging_preference'])
    with col14:
        health = st.selectbox("Health Concerns", feature_options['health_concerns'])
    with col15:
        situation = st.selectbox("Typical Consumption Situations", feature_options['typical_consumption_situations'])

    submit = st.form_submit_button("🔮 Calculate Price Range")

# Prediction logic
if submit:
    try:
        input_dict = {
            'age_group': label_encodings['age_group'][age_group],
            'income_levels': label_encodings['income_levels'][income],
            'health_concerns': label_encodings['health_concerns'][health],
            'consume_frequency(weekly)': label_encodings['consume_frequency(weekly)'][freq],
            'preferable_consumption_size': label_encodings['preferable_consumption_size'][size],
            'cf_ab_score': cf_ab_score,
            'zas_score': zas_score,
            'BSI': bsi_score,
            'gender': gender,
            'zone': zone,
            'occupation': occupation,
            'flavor_preference': flavor,
            'typical_consumption_situations': situation,
            'current_brand': brand,
            'purchase_channel': channel,
            'packaging_preference': packaging,
            'awareness_of_other_brands': awareness,
            'reasons_for_choosing_brands': reason
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encoding for string columns
        cat_cols = input_df.select_dtypes(include='object').columns.tolist()
        input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # Align to expected model input columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Predict
        pred_encoded = model.predict(input_df)[0]
        final_prediction = label_encoder_y.inverse_transform([pred_encoded])[0]

        # Confidence
        try:
            proba = model.predict_proba(input_df)[0]
            confidence = np.max(proba) * 100
        except:
            confidence = None

        # Output
        st.success(f"🎯 Predicted Price Range: {final_prediction}")
        if confidence:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
