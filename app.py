import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
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

# Load model from local file
@st.cache_resource
def load_model():
    try:
        import pickle
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
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
    """
    Categorize readmission risk based on model probability output.

    Thresholds based on clinical utility and model performance:
    - Low: < 40% (below typical readmission baseline)
    - Medium: 40-50% (moderate risk, warrants attention)
    - High: > 50% (more likely to be readmitted than not)
    """
    if probability < 0.40:
        return "Low", "#9AD66B", "low-risk"
    elif probability < 0.50:
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

# Helper function to identify key contributing factors
def get_key_factors(inputs, input_df):
    """Identify key factors that likely influenced the prediction"""
    factors = []

    # Check for elderly status
    if inputs['age'] in ['[70-80)', '[80-90)', '[90-100)']:
        factors.append("**Advanced age** - Patients 70+ have increased readmission risk")

    # Check for high prior utilization
    total_visits = inputs['number_emergency'] + inputs['number_inpatient'] + inputs['number_outpatient']
    if total_visits >= 2:
        factors.append(f"**High prior utilization** - {total_visits} prior visits in past year indicates healthcare needs")

    # Check for emergency history
    if inputs['number_emergency'] > 0:
        factors.append(f"**Emergency department history** - {inputs['number_emergency']} ED visit(s) in past year")

    # Check for inpatient history
    if inputs['number_inpatient'] > 0:
        factors.append(f"**Previous hospitalizations** - {inputs['number_inpatient']} inpatient stay(s) in past year")

    # Check for medication complexity
    if inputs['num_medications'] > 15:
        factors.append(f"**High medication burden** - {inputs['num_medications']} medications increases complexity")

    # Check for medication changes
    if inputs['change'] == 'Yes':
        factors.append("**Recent medication changes** - Adjustments during hospitalization require monitoring")

    # Check for insulin use
    if inputs['insulin'] not in ['No', 'no']:
        factors.append("**Insulin therapy** - Requires careful management and patient education")

    # Check for long hospital stay
    if inputs['time_in_hospital'] > 7:
        factors.append(f"**Extended hospital stay** - {inputs['time_in_hospital']} days indicates complex medical needs")

    # Check for multiple diagnoses
    if inputs['number_diagnoses'] > 7:
        factors.append(f"**Multiple diagnoses** - {inputs['number_diagnoses']} conditions increases care complexity")

    # Check for high procedure count
    if inputs['num_lab_procedures'] > 60:
        factors.append(f"**Extensive testing** - {inputs['num_lab_procedures']} lab procedures suggests complex case")

    # Add baseline explanation if no major risk factors found
    if not factors:
        # Check for moderate age
        if inputs['age'] in ['[50-60)', '[60-70)']:
            factors.append(f"**Age {inputs['age']}** - Moderate age group with baseline diabetes risk")

        # Check for diabetes diagnosis codes
        if inputs['diag_1'] == '250' or inputs['diag_2'] == '250' or inputs['diag_3'] == '250':
            factors.append("**Diabetes diagnosis** - Primary condition requires ongoing management")

        # Check if on diabetes medication
        if inputs['diabetesMed'] == 'Yes':
            factors.append("**Active diabetes treatment** - Patient requires medication management")

    return factors

def prepare_input_data(inputs):
    """Prepare input data with EXACT 58 features the model expects"""

    # Convert Yes/No to Ch/No for medication changes (model expects 'Ch')
    change_value = 'Ch' if inputs.get('change', 'No') == 'Yes' else 'No'

    # Create dataframe with all 58 columns in exact order from training
    data = {
        'race': inputs.get('race', 'Caucasian'),
        'gender': inputs.get('gender', 'Male'),
        'age': inputs.get('age', '[70-80)'),
        'admission_type_id': inputs.get('admission_type_id', 1),
        'discharge_disposition_id': inputs.get('discharge_disposition_id', 1),
        'admission_source_id': inputs.get('admission_source_id', 1),
        'time_in_hospital': inputs.get('time_in_hospital', 3),
        'num_lab_procedures': inputs.get('num_lab_procedures', 40),
        'num_procedures': inputs.get('num_procedures', 0),
        'num_medications': inputs.get('num_medications', 15),
        'number_outpatient': inputs.get('number_outpatient', 0),
        'number_emergency': inputs.get('number_emergency', 0),
        'number_inpatient': inputs.get('number_inpatient', 0),
        'diag_1': inputs.get('diag_1', '250'),
        'diag_2': inputs.get('diag_2', '250'),
        'diag_3': inputs.get('diag_3', '250'),
        'number_diagnoses': inputs.get('number_diagnoses', 7),
        'metformin': inputs.get('metformin', 'No'),
        'repaglinide': 'No',
        'nateglinide': 'No',
        'chlorpropamide': 'No',
        'glimepiride': 'No',
        'acetohexamide': 'No',
        'glipizide': 'No',
        'glyburide': 'No',
        'tolbutamide': 'No',
        'pioglitazone': 'No',
        'rosiglitazone': 'No',
        'acarbose': 'No',
        'miglitol': 'No',
        'troglitazone': 'No',
        'tolazamide': 'No',
        'examide': 'No',
        'citoglipton': 'No',
        'insulin': inputs.get('insulin', 'No'),
        'glyburide-metformin': 'No',
        'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No',
        'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No',
        'change': change_value,
        'diabetesMed': inputs.get('diabetesMed', 'Yes'),
    }
    
    df = pd.DataFrame([data])
    
    # Calculate median values for thresholds (approximations from typical dataset)
    num_meds_median = 15
    num_diagnoses_median = 7
    total_visits_q75 = 2
    time_in_hosp_median = 3
    
    # Encode age to numeric
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
               '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
               '[80-90)': 85, '[90-100)': 95}
    df['age_numeric'] = df['age'].map(age_map)
    
    # Calculate engineered features EXACTLY as in training
    medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                       'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                       'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                       'metformin-pioglitazone']
    
    df['total_medications'] = sum(1 for col in medication_cols if df[col].iloc[0] not in ['No', 'no'])
    df['medication_changes'] = sum(1 for col in medication_cols if df[col].iloc[0] in ['Up', 'Down'])
    df['insulin_prescribed'] = int(df['insulin'].iloc[0] not in ['No', 'no'])
    df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures']
    df['procedures_per_day'] = df['total_procedures'] / (df['time_in_hospital'] + 1)
    df['high_medication_load'] = (df['num_medications'] > num_meds_median).astype(int)
    df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['has_emergency_history'] = (df['number_emergency'] > 0).astype(int)
    df['has_inpatient_history'] = (df['number_inpatient'] > 0).astype(int)
    df['multiple_diagnoses'] = (df['number_diagnoses'] > num_diagnoses_median).astype(int)
    df['is_elderly'] = (df['age_numeric'] >= 70).astype(int)
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    df['elderly_emergency_risk'] = df['is_elderly'] * df['has_emergency_history']
    df['complex_case'] = df['high_medication_load'] * df['multiple_diagnoses']
    df['high_utilization'] = ((df['total_prior_visits'] > total_visits_q75) & 
                              (df['time_in_hospital'] > time_in_hosp_median)).astype(int)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3'] + medication_cols + ['change', 'diabetesMed']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Return in EXACT column order from training
    column_order = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
                    'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
                    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'metformin', 'repaglinide', 
                    'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 
                    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
                    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 
                    'total_medications', 'medication_changes', 'insulin_prescribed', 'total_procedures', 
                    'procedures_per_day', 'high_medication_load', 'total_prior_visits', 
                    'has_emergency_history', 'has_inpatient_history', 'multiple_diagnoses', 
                    'age_numeric', 'is_elderly', 'medication_changed', 'elderly_emergency_risk', 
                    'complex_case', 'high_utilization']
    
    return df[column_order]

if page == "Score a Patient":
    st.markdown("<h2>Patient Risk Scoring</h2>", unsafe_allow_html=True)
    
    if error:
        st.error(f"⚠️ **Model loading failed**: {error}")
        st.info("Make sure you've added DATABRICKS_HOST and DATABRICKS_TOKEN to Streamlit secrets")
    elif model is None:
        st.warning("⚠️ **Model not loaded**. Check your Databricks connection.")
    else:
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
            # Admission Type mapping
            admission_types = {
                "Emergency": 1,
                "Urgent": 2,
                "Elective": 3,
                "Newborn": 4,
                "Not Available": 5,
                "NULL": 6,
                "Trauma Center": 7,
                "Not Mapped": 8
            }
            admission_type = st.selectbox("Admission Type", list(admission_types.keys()), index=0)
            admission_type_id = admission_types[admission_type]

            # Discharge Disposition mapping
            discharge_dispositions = {
                "Discharged to home": 1,
                "Discharged/transferred to another short term hospital": 2,
                "Discharged/transferred to SNF (skilled nursing facility)": 3,
                "Discharged/transferred to ICF (intermediate care facility)": 4,
                "Discharged/transferred to another type of inpatient care": 5,
                "Discharged/transferred to home with home health service": 6,
                "Left AMA (against medical advice)": 7,
                "Discharged/transferred to home under care of Home IV provider": 8,
                "Admitted as an inpatient to this hospital": 9,
                "Neonate discharged to another hospital for neonatal aftercare": 10,
                "Expired": 11,
                "Still patient or expected to return for outpatient services": 12,
                "Hospice / home": 13,
                "Hospice / medical facility": 14,
                "Discharged/transferred within this institution to Medicare approved swing bed": 15,
                "Discharged/transferred/referred another institution for outpatient services": 16,
                "Discharged/transferred/referred to this institution for outpatient services": 17,
                "NULL": 18,
                "Expired at home. Medicaid only, hospice.": 19,
                "Expired in a medical facility. Medicaid only, hospice.": 20,
                "Expired, place unknown. Medicaid only, hospice.": 21,
                "Discharged/transferred to another rehab fac": 22,
                "Discharged/transferred to a long term care hospital": 23,
                "Discharged/transferred to a nursing facility certified under Medicaid": 24,
                "Not Mapped": 25,
                "Unknown/Invalid": 26,
                "Discharged/transferred to a federal health care facility": 27,
                "Discharged/transferred/referred to a psychiatric hospital": 28,
                "Discharged/transferred to a Critical Access Hospital (CAH)": 29
            }
            discharge_disposition = st.selectbox("Discharge Disposition",
                                                list(discharge_dispositions.keys()), index=0)
            discharge_disposition_id = discharge_dispositions[discharge_disposition]

            # Admission Source mapping
            admission_sources = {
                "Physician Referral": 1,
                "Clinic Referral": 2,
                "HMO Referral": 3,
                "Transfer from a hospital": 4,
                "Transfer from a Skilled Nursing Facility (SNF)": 5,
                "Transfer from another health care facility": 6,
                "Emergency Room": 7,
                "Court/Law Enforcement": 8,
                "Not Available": 9,
                "Transfer from critical access hospital": 10,
                "Normal Delivery": 11,
                "Premature Delivery": 12,
                "Sick Baby": 13,
                "Extramural Birth": 14,
                "Not Available": 15,
                "NULL": 17,
                "Transfer From Another Home Health Agency": 18,
                "Readmission to Same Home Health Agency": 19,
                "Not Mapped": 20,
                "Unknown/Invalid": 21,
                "Transfer from hospital inpatient in the same facility": 22,
                "Born inside this hospital": 23,
                "Born outside this hospital": 24,
                "Transfer from Ambulatory Surgery Center": 25
            }
            admission_source = st.selectbox("Admission Source",
                                           list(admission_sources.keys()), index=6)  # Default to Emergency Room
            admission_source_id = admission_sources[admission_source]
        
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
        with st.expander("Medications", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                change = st.selectbox("Medication Changed During Stay", ['No', 'Yes'])
                diabetesMed = st.selectbox("Diabetes Med Prescribed", ['Yes', 'No'])
            
            with col4:
                st.markdown("**Key Medications**")
                metformin = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])
                insulin = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
        
        # Diagnosis codes (optional - defaults to diabetes)
        with st.expander("Diagnosis Codes (Optional - ICD-9)", expanded=False):
            st.info("💡 Diagnosis codes are optional. Default is 250 (Diabetes). Common codes: 250 (Diabetes), 401 (Hypertension), 428 (Heart Failure), 427 (Cardiac Arrhythmia)")
            diag_1 = st.text_input("Primary Diagnosis Code", value="250")
            diag_2 = st.text_input("Secondary Diagnosis Code", value="250")
            diag_3 = st.text_input("Tertiary Diagnosis Code", value="250")
        
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
                
                # Convert to numpy array with float64 dtype (XGBoost requirement)
                input_array = input_df.values.astype('float64')
                
                # Get prediction probability
                prediction_proba = model.predict_proba(input_array)

                # Get probability of positive class (readmission = 1)
                probability = float(prediction_proba[0][1])

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
                            {'range': [0, 40], 'color': '#E8F5E9'},
                            {'range': [40, 50], 'color': '#FFF3E0'},
                            {'range': [50, 100], 'color': '#FFEBEE'}
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
                
                # Key factors influencing prediction
                st.markdown("<h3>Key Factors Contributing to This Risk Assessment</h3>", unsafe_allow_html=True)
                key_factors = get_key_factors(inputs, input_df)

                if key_factors:
                    st.info("The following patient characteristics likely influenced this risk prediction:")
                    for factor in key_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.info("• Patient profile shows standard risk characteristics with no major red flags")

                # Recommended interventions
                st.markdown("<h3>Suggested Next Steps</h3>", unsafe_allow_html=True)
                interventions = get_interventions(risk_level)
                for intervention in interventions:
                    st.markdown(f"- {intervention}")

            except Exception as e:
                st.error(f"⚠️ **Prediction failed**: {str(e)}")
                st.info("This may be due to feature mismatch. Check that input features match training data.")

        # Model info and disclaimer at bottom of Score a Patient page
        st.markdown("---")
        st.success("✅ Model loaded successfully - XGBoost trained on 100,000+ patient records")
        st.markdown("---")
        st.warning("""
            **⚠️ EDUCATIONAL PROJECT DISCLAIMER**

            This application is a school assignment created for educational purposes only.
            The predictions and recommendations provided by this tool should **NOT** be used for actual medical decision-making
            or patient care. This is not a substitute for professional medical advice, diagnosis, or treatment.
            Always consult qualified healthcare professionals for medical decisions.
        """)

elif page == "About the Model":
    st.markdown("<h2>About the Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### The Challenge
    
    Hospitals lose millions annually on preventable readmissions of diabetes patients. 
    GlucoBridge uses machine learning trained on 10 years of data from 130 US hospitals to predict 
    30-day readmission risk, enabling care teams to focus resources on the patients who need them most.
    
    ### Model Development Journey
    
    **What We Tried:**
    - **9 Different Algorithms**: Logistic Regression, Decision Trees, Random Forest, XGBoost, 
      Gradient Boosting, k-NN, Naive Bayes, SVM, Neural Networks, and Ensemble methods
    - **Hyperparameter Tuning**: 2-3 parameters per model with multiple values, resulting in 
      100+ model runs logged to MLflow
    - **Class Imbalance Handling**: Initially used SMOTE (Synthetic Minority Over-sampling Technique)
    
    **Challenges Encountered:**
    - **Data Cleaning Misstep**: Initial aggressive cleaning reduced dataset from 100k+ rows to just 
      200 rows, requiring complete pipeline restart
    - **Training Time Issues**: Neural Network and SVM models took 2+ hours per run, making iteration 
      impractical under project deadlines  
    - **SMOTE Problem**: Created synthetic training data that caused the model to predict >95% risk 
      for nearly all patients—clinically useless and would overwhelm care teams with false alarms
    - **Model Registration Failures**: Initial MLflow logging didn't include proper signatures, 
      preventing model registration in Databricks Unity Catalog
    - **Feature Mismatch in Deployment**: Trained model expected 58 specific features in exact order; 
      deployment required careful feature engineering pipeline to match training preprocessing
    
    **The Solution:**
    - **XGBoost with scale_pos_weight**: Properly handles class imbalance without synthetic data, 
      trains in 12.5 seconds vs 2+ hours for alternatives
    - **Streamlined Preprocessing**: Rebuilt pipeline to preserve data while handling missing values 
      appropriately (dropped only features >40% missing like A1C results)
    - **Feature Engineering**: Created 15+ derived features (medication changes, elderly risk scores, 
      utilization patterns) that improved model performance
    - **Real-World Calibration**: Model now produces realistic risk distributions (9-35% range)
      that align with clinical expectations
    - **Production-Ready Deployment**: Proper MLflow signatures enable seamless model registry 
      and web application integration
    
    ### Final Model Performance
    """)
    
    # Model comparison table with actual metrics - BEST MODEL FIRST
    comparison_data = {
        'Model': ['XGBoost (Final)', 'Random Forest', 'Logistic Regression'],
        'F1 Score': [0.63, 0.60, 0.58],
        'Accuracy': [0.65, 0.64, 0.62],
        'Recall': [0.64, 0.57, 0.55],
        'ROC AUC': [0.70, 0.71, 0.68],
        'Notes': [
            '✓ Best balance - optimized for recall',
            'Better performance, less interpretable',
            'Simple baseline, interpretable'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Why This Model Won:**
    - **Highest Recall (64%)**: Catches more high-risk patients, critical in healthcare where 
      missing a readmission is costly
    - **Balanced Performance**: Strong F1 score (63%) shows good precision-recall tradeoff
    - **Realistic Predictions**: Risk scores range appropriately (not clustering at extremes)
    - **Fast Training**: 12.5 seconds enables rapid iteration
    
    ### Training Data
    
    - **Source**: 130 US hospitals (1999-2008)
    - **Records**: ~100,000 inpatient encounters
    - **Features**: 58 features after engineering (clinical, demographic, medication, utilization)
    - **Target**: 30-day readmission (any cause)
    - **Class Distribution**: Imbalanced (~45% readmission rate after cleaning)
    
    ### Technical Implementation
    
    **Model Details:**
    - **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
    - **Hyperparameters**: 
        - n_estimators=100
        - max_depth=5
        - learning_rate=0.3
        - scale_pos_weight=1.17 (handles class imbalance)
    - **Validation**: Stratified train-test split (80/20)
    - **Training Time**: 12.5 seconds
    
    **Deployment Pipeline:**
    - **Experimentation**: 100+ runs tracked in Databricks MLflow
    - **Model Registry**: Registered in Unity Catalog (`workspace.default.diabetes_readmission_2`)
    - **Deployment**: Streamlit Cloud with automated CI/CD from GitHub
    - **Monitoring**: Model versioning enables A/B testing and rollback
    
    ### Feature Engineering Highlights
    
    Key engineered features that improved performance:
    - **Medication complexity**: Count of active medications, recent changes
    - **Utilization patterns**: Prior emergency visits, inpatient history, high utilization flag
    - **Risk interactions**: Elderly + emergency history, complex case indicators
    - **Procedure intensity**: Procedures per day, total procedures
    
    ### Real-World Integration
    
    This model can be integrated into:
    - **Electronic Health Record (EHR) systems**: Real-time scoring at discharge
    - **Case management workflows**: Prioritize high-risk patients for follow-up
    - **Population health dashboards**: Track readmission trends and intervention effectiveness
    - **Care coordination tools**: Automated referrals and appointment scheduling
    
    ### Lessons Learned
    
    1. **Class imbalance requires careful handling**: Synthetic oversampling (SMOTE) can create 
       unrealistic predictions; use proper weighting instead
    2. **Feature engineering matters**: Domain-specific features (medication patterns, utilization) 
       outperformed raw clinical codes
    3. **Model interpretability vs. performance**: XGBoost provided the best tradeoff—better than 
       simple models, more interpretable than deep learning
    4. **Deployment requires the full pipeline**: Model alone isn't enough; need proper preprocessing, 
       feature calculation, and error handling in production
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <p><strong>GlucoBridge Health</strong> | Clinical Analytics Platform</p>
        <p style='font-size: 0.9rem;'>Deployed via Streamlit Cloud | Tracked in Databricks MLflow | Code on GitHub</p>
    </div>
    """, unsafe_allow_html=True)

    # Author Information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem;'>
        <h3 style='color: #0B1F33; margin-bottom: 1rem;'>About the Author</h3>
        <p><strong>Chandra Carr</strong></p>
        <p>Graduate Student, W.P. Carey School of Business</p>
        <p>Arizona State University</p>
        <p style='margin-top: 1rem;'>
            📧 <a href='mailto:cgcarr@asu.edu'>cgcarr@asu.edu</a><br>
            💼 <a href='https://www.linkedin.com/in/chandragcarr/' target='_blank'>LinkedIn Profile</a><br>
            💻 <a href='https://github.com/WhiteRabbit-glitch' target='_blank'>GitHub</a>
        </p>
        <p style='font-size: 0.9rem; margin-top: 1rem; color: #4A4F5C;'>
            CIS 508 - Machine Learning in Business<br>
            December 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Educational Disclaimer at bottom of About the Model page
    st.markdown("---")
    st.warning("""
        **⚠️ EDUCATIONAL PROJECT DISCLAIMER**

        This application is a school assignment created for educational purposes only.
        The predictions and recommendations provided by this tool should **NOT** be used for actual medical decision-making
        or patient care. This is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult qualified healthcare professionals for medical decisions.
    """)
