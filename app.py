import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard", 
    page_icon="⚙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4CAF50, #F44336);
    }
</style>
""", unsafe_allow_html=True)

# Load pipeline models
@st.cache_resource
def load_models():
    binary_pipeline = joblib.load('models/binary_pipeline.pkl')
    failure_type_pipeline = joblib.load('models/failure_type_pipeline.pkl')
    return binary_pipeline, failure_type_pipeline

binary_pipeline, failure_type_pipeline = load_models()

# Header
st.markdown('<p class="main-header">⚙️ Intelligent Machine Health Monitor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time predictive maintenance dashboard using XGBoost & Random Forest</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043004.png", width=100)
    st.title("Sensor Inputs")
    st.markdown("Adjust the telemetry data below to simulate machine conditions.")
    
    type_input = st.selectbox("Machine Type (Quality)", ["L (Low)", "M (Medium)", "H (High)"])[0]
    air_temp = st.slider("Air temperature [K]", 290.0, 310.0, 298.1, 0.1)
    process_temp = st.slider("Process temperature [K]", 300.0, 320.0, 308.6, 0.1)
    rotational_speed = st.slider("Rotational speed [rpm]", 1000.0, 3000.0, 1551.0, 1.0)
    torque = st.slider("Torque [Nm]", 10.0, 100.0, 42.8, 0.1)
    tool_wear = st.slider("Tool wear [min]", 0.0, 300.0, 0.0, 1.0)
    
    predict_btn = st.button("Run Diagnostic Check 🚀", use_container_width=True, type="primary")

# Main Dashboard layout
st.markdown("### 📊 Current Telemetry")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Machine Type", type_input)
col2.metric("Air Temp", f"{air_temp} K")
col3.metric("Process Temp", f"{process_temp} K")
col4.metric("Speed", f"{rotational_speed} rpm")
col5.metric("Torque", f"{torque} Nm")

st.divider()

if predict_btn:
    with st.spinner("Analyzing sensor data..."):
        # Feature Engineering
        temp_diff = process_temp - air_temp
        mechanical_power = rotational_speed * torque
        wear_ratio = tool_wear / (rotational_speed + 1e-5)
        torque_speed_ratio = torque / (rotational_speed + 1e-5)
        normalized_tool_wear = tool_wear / 253.0
        
        feature_names = [
            'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'Temperature Difference', 'Mechanical Power',
            'Wear Ratio', 'Torque-Speed Ratio', 'Normalized Tool Wear'
        ]
        
        features = [
            type_input, air_temp, process_temp, rotational_speed, torque, tool_wear,
            temp_diff, mechanical_power, wear_ratio, torque_speed_ratio, normalized_tool_wear
        ]
        
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Binary Prediction
        try:
            binary_pred = int(binary_pipeline.predict(features_df)[0])
            binary_prob = float(binary_pipeline.predict_proba(features_df)[0][1])
        except AttributeError:
            import xgboost as xgb
            booster = binary_pipeline.named_steps['classifier'].get_booster()
            processed_features = binary_pipeline.named_steps['preprocessor'].transform(features_df)
            probs = booster.predict(xgb.DMatrix(processed_features))
            binary_prob = float(probs[0])
            binary_pred = int(binary_prob > 0.5)
            
        # UI Display for Results
        st.markdown("### 🩺 Diagnostic Report")
        
        if binary_pred == 0:
            st.success("✅ **STATUS: OPTIMAL** - The machine is operating normally.")
            st.info(f"**Failure Risk:** {binary_prob:.2%}")
            st.progress(1.0 - binary_prob, text="Health Score")
        else:
            st.error(f"⚠️ **STATUS: CRITICAL** - Machine failure predicted!")
            st.warning(f"**Failure Risk:** {binary_prob:.2%}")
            st.progress(binary_prob, text="Risk Level")
            
            # Predict Failure Type
            fail_type = failure_type_pipeline.predict(features_df)[0]
            
            st.markdown("---")
            st.markdown(f"#### 🔍 Predicted Failure Mode: **{fail_type}**")
            
            # Recommendations
            if fail_type == 'TWF':
                st.error("🛠️ **Action Required:** Tool Wear Failure detected. Replace the cutting tool immediately and check lubrication.")
            elif fail_type == 'HDF':
                st.error("❄️ **Action Required:** Heat Dissipation Failure detected. Inspect the cooling system, check for blockages, and ensure ambient temperature is controlled.")
            elif fail_type == 'PWF':
                st.error("⚡ **Action Required:** Power Failure detected. Check power supply stability, motor voltage, and electrical connections.")
            elif fail_type == 'OSF':
                st.error("⚖️ **Action Required:** Overstrain Failure detected. Reduce the machine load, check for material inconsistencies, and verify torque limits.")
            elif fail_type == 'RNF':
                st.error("🎲 **Action Required:** Random Failure predicted. Perform a comprehensive full-system diagnostic check.")
            else:
                st.error("🚨 **Action Required:** Unclassified failure. Halt operations and perform standard maintenance checks.")

        # Extra metrics
        with st.expander("Show Advanced Diagnostics"):
            st.write("Derived Features calculated by the Machine Learning Engine:")
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            adv_col1.metric("Temp Difference", f"{temp_diff:.2f} K")
            adv_col2.metric("Mechanical Power", f"{mechanical_power:.2f} W")
            adv_col3.metric("Tool Wear Level", f"{tool_wear} min")
else:
    st.info("👈 Adjust the parameters in the sidebar and click **Run Diagnostic Check** to see the prediction.")
