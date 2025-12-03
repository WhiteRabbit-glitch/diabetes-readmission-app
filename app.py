import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import os

# Page config
st.set_page_config(
    page_title="GlucoBridge Readmission Risk Viewer",
    page_icon="🌉",
    layout="wide"
)

# Custom CSS for GlucoBridge branding
st.markdown("""
    <style>
    .main {
        background-color: #F4F6F8;
    }
    .stButton>button {
        background-color: #1B9AAA;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover {
        background-color: #157885;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #0B1F33;
        font-family: 'Poppins', sans-serif;
    }
    .risk-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .high-risk {
        border-left: 5px solid #FF6B5A;
    }
    .medium-risk {
        border-left: 5px solid #FFB84D;
    }
    .low-risk {
        border-left: 5px solid #9AD66B;
    }
    </style>
""", unsafe_allow_html=True)

# Logo and header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #0B1F33; font-size: 2.5rem; margin-bottom: 0.5rem;'>
            🌉 <span style='color: #1B9AAA;'>GlucoBridge</span> Health
        </h1>
        <p style='color: #4A4F5C; font-size: 1.2rem;'>Bridging diabetes care from hospital to home</p>
    </div>
""", unsafe_allow_html=True)

# Databricks connection and model loading
@st.cache_resource
def load_model():
    try:
        # Check secrets exist
        if "DATABRICKS_HOST" not in st.secrets:
            return None, "DATABRICKS_HOST not found in secrets"
        if "DATABRICKS_TOKEN" not in st.secrets:
            return None, "DATABRICKS_TOKEN not found in secrets"
            
        os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
        os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]
        mlflow.set_tracking_uri("databricks")
        
        # Try to load with timeout handling
        st.info(f"Connecting to: {st.secrets['DATABRICKS_HOST']}")
        model_uri = "models:/workspace.default.diabetes_readmission/1"
        st.info(f"Loading model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        return model, None
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_detail

model, error = load_model()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Score a Patient", "About the Model"])

# Helper function to calculate risk category
def get_risk_category(probability):
    if probability < 0.15:
        return "Low", "#9AD66B", "low-risk"
    elif probability < 0.30:
        return "Medium", "#FFB84D", "medium-risk"
    else:
        return "High", "#FF6B5A", "high-risk"

# Helper function to get interventions
def get_interventions(risk_level):
    if risk_level == "Low":
        return [
            "✓ Standard discharge instructions",
            "✓ Routine follow-up appointment within 4–6 weeks"
        ]
    elif risk_level == "Medium":
        return [
            "⚠ Schedule diabetes educator visit before discharge",
            "⚠ Arrange follow-up call within 7 days",
            "⚠ Ensure A1C result communicated to primary care"
        ]
    else:
        return [
            "🚨 Flag for case manager review before discharge",
            "🚨 Schedule follow-up clinic appointment within 3–5 days",
            "🚨 Consider home health referral",
            "🚨 Confirm medication reconciliation and affordability"
        ]

def prepare_input_data(inputs):
    """Prepare input data in the format the model expects"""
    # This needs to match the exact features your model was trained on
    # You'll need to adjust this based on your actual model's expected input
    
    # Create a dataframe with all required features
    df = pd.DataFrame([inputs])
    
    # Encode age to numeric
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
               '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
               '[80-90)': 85, '[90-100)': 95}
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(age_map)
    
    # Add engineered features that your model expects
    df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures']
    df['procedures_per_day'] = df['total_procedures'] / (df['time_in_hospital'] + 1)
    df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['has_emergency_history'] = (df['number_emergency'] > 0).astype(int)
    df['has_inpatient_history'] = (df['number_inpatient'] > 0).astype(int)
    df['high_medication_load'] = (df['num_medications'] > 15).astype(int)  # Using median=15 as approximation
    df['multiple_diagnoses'] = (df['number_diagnoses'] > 7).astype(int)
    df['is_elderly'] = (df['age_numeric'] >= 70).astype(int)
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    df['elderly_emergency_risk'] = df['is_elderly'] * df['has_emergency_history']
    df['complex_case'] = df['high_medication_load'] * df['multiple_diagnoses']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

if page == "Score a Patient":
    st.markdown("<h2>Patient Risk Scoring</h2>", unsafe_allow_html=True)
    
    if error:
        st.error(f"⚠️ **Model loading failed**: {error}")
        st.info("Make sure you've added DATABRICKS_HOST and DATABRICKS_TOKEN to Streamlit secrets")
    elif model is None:
        st.warning("⚠️ **Model not loaded**. Check your Databricks connection.")
    else:
        st.success("✅ Model loaded successfully from Databricks")
        st.info("📋 Enter patient information to calculate 30-day readmission risk")
        
        # Create form in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demographics & Stay Details")
            age = st.selectbox("Age Range", 
                             ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                              '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                             index=6)
            race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
            time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=3)
            
            st.markdown("### Admission Details")
            admission_type_id = st.selectbox("Admission Type", list(range(1, 9)), index=0)
            discharge_disposition_id = st.selectbox("Discharge Disposition", list(range(1, 30)), index=0)
            admission_source_id = st.selectbox("Admission Source", list(range(1, 26)), index=0)
        
        with col2:
            st.markdown("### Clinical Information")
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=132, value=40)
            num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=6, value=0)
            num_medications = st.number_input("Number of Medications", min_value=1, max_value=81, value=15)
            number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=7)
            
            st.markdown("### Prior Healthcare Utilization")
            number_outpatient = st.number_input("Outpatient Visits (past year)", min_value=0, max_value=42, value=0)
            number_emergency = st.number_input("Emergency Visits (past year)", min_value=0, max_value=76, value=0)
            number_inpatient = st.number_input("Inpatient Visits (past year)", min_value=0, max_value=21, value=0)
        
        # Additional features
        with st.expander("Lab Results & Medications", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                max_glu_serum = st.selectbox("Max Glucose Serum", ['None', '>200', '>300', 'Norm'])
                A1Cresult = st.selectbox("A1C Result", ['None', '>7', '>8', 'Norm'])
                change = st.selectbox("Medication Change", ['No', 'Ch'])
                diabetesMed = st.selectbox("Diabetes Med Prescribed", ['Yes', 'No'])
            
            with col4:
                st.markdown("**Key Medications**")
                metformin = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])
                insulin = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
        
        # Diagnosis codes
        with st.expander("Diagnosis Codes (ICD-9)", expanded=False):
            diag_1 = st.text_input("Primary Diagnosis", value="250")
            diag_2 = st.text_input("Secondary Diagnosis", value="250")
            diag_3 = st.text_input("Tertiary Diagnosis", value="250")
        
        if st.button("🎯 Calculate Risk Score", use_container_width=True):
            # Prepare inputs
            inputs = {
                'age': age,
                'race': race,
                'gender': gender,
                'time_in_hospital': time_in_hospital,
                'admission_type_id': admission_type_id,
                'discharge_disposition_id': discharge_disposition_id,
                'admission_source_id': admission_source_id,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_diagnoses': number_diagnoses,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'max_glu_serum': max_glu_serum,
                'A1Cresult': A1Cresult,
                'change': change,
                'diabetesMed': diabetesMed,
                'metformin': metformin,
                'insulin': insulin,
                'diag_1': diag_1,
                'diag_2': diag_2,
                'diag_3': diag_3
            }
            
            try:
                # Prepare data
                input_df = prepare_input_data(inputs)
                
                # Get prediction
                prediction = model.predict(input_df)
                probability = prediction[0] if isinstance(prediction[0], float) else prediction[0][1]
                
                risk_level, risk_color, risk_class = get_risk_category(probability)
                
                # Display results
                st.markdown("---")
                st.markdown("<h2>Risk Assessment Results</h2>", unsafe_allow_html=True)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "30-Day Readmission Risk", 'font': {'size': 24, 'color': '#0B1F33'}},
                    number={'suffix': "%", 'font': {'size': 48, 'color': risk_color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4A4F5C"},
                        'bar': {'color': risk_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#4A4F5C",
                        'steps': [
                            {'range': [0, 15], 'color': '#E8F5E9'},
                            {'range': [15, 30], 'color': '#FFF3E0'},
                            {'range': [30, 100], 'color': '#FFEBEE'}
                        ],
                        'threshold': {
                            'line': {'color': "#0B1F33", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=80, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk category card
                st.markdown(f"""
                    <div class='risk-card {risk_class}'>
                        <h3 style='margin-top: 0;'>Risk Level: {risk_level}</h3>
                        <p style='font-size: 1.1rem; color: #4A4F5C;'>
                            Predicted 30-day readmission probability: <strong>{probability*100:.1f}%</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Recommended interventions
                st.markdown("<h3>Suggested Next Steps</h3>", unsafe_allow_html=True)
                interventions = get_interventions(risk_level)
                for intervention in interventions:
                    st.markdown(f"- {intervention}")
                    
            except Exception as e:
                st.error(f"⚠️ **Prediction failed**: {str(e)}")
                st.info("This may be due to feature mismatch. Check that input features match training data.")

elif page == "About the Model":
    st.markdown("<h2>About the Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### The Challenge
    
    Hospitals lose millions of dollars every year on preventable readmissions of patients with diabetes. 
    GlucoBridge Health uses machine learning trained on ten years of inpatient encounters from 130 US 
    hospitals to predict which diabetes patients are most likely to be readmitted within 30 days of discharge.
    
    ### Our Solution
    
    The goal is to give care teams a real-time risk score and concrete intervention suggestions at the 
    moment of discharge, so limited resources can be focused on the patients who need them most.
    
    ### Model Performance
    """)
    
    # Model comparison table
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (Final)'],
        'F1 Score': [0.58, 0.60, 0.61],
        'Accuracy': [0.62, 0.64, 0.65],
        'Recall': [0.55, 0.57, 0.58],
        'Notes': [
            'Simple baseline, interpretable',
            'Better performance, less interpretable',
            'Best trade-off between metrics'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### Training Data
    
    - **Source**: 130 US hospitals
    - **Timeframe**: 10 years of inpatient encounters (1999-2008)
    - **Features**: 50+ clinical and demographic variables
    - **Outcome**: 30-day readmission (any cause)
    
    ### Model Details
    
    - **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
    - **Hyperparameters**: n_estimators=100, max_depth=5, learning_rate=0.3
    - **Training approach**: SMOTE oversampling to handle class imbalance
    - **Validation**: Stratified train-test split (80/20)
    - **Registered**: Databricks MLflow Model Registry
    
    ### Integration
    
    This model is logged in MLflow and can be integrated into:
    - Electronic Health Record (EHR) systems
    - Case management workflows
    - Population health dashboards
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #4A4F5C;'>
        <p><strong>GlucoBridge Health</strong> | Clinical Analytics Platform</p>
        <p style='font-size: 0.9rem;'>For questions or support, contact your system administrator</p>
    </div>
    """, unsafe_allow_html=True)
