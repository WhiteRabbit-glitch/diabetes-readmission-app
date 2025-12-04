# 🌉 GlucoBridge Health: Hospital Readmission Risk Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Bridging diabetes care from hospital to home through predictive analytics**

End-to-end machine learning project predicting hospital readmission risk for diabetic patients. Developed as academic coursework (CIS 508 - Machine Learning in Business, ASU), demonstrating real-world ML pipeline development including data preprocessing challenges, extensive model experimentation (490+ MLflow runs), class imbalance handling, and production deployment.

**Live Demo:** [GlucoBridge Health Application](https://your-app-url.streamlit.app)

---

## 📊 Business Problem

### The Challenge
- **$41 billion** lost annually to hospital readmissions in the US
- **20-30%** of diabetic patients readmitted within 30 days
- Limited care team resources require risk-based prioritization
- Traditional clinical judgment alone misses high-risk patients

### The Solution
ML-powered risk stratification at discharge to:
- **Predict readmission risk** with 70% AUC
- **Stratify patients** into Low/Medium/High risk tiers
- **Recommend interventions** based on risk level
- **Enable resource allocation** to highest-need patients

---

## 🎯 Model Performance

**Production Model:** XGBoost trained with `scale_pos_weight` for class imbalance (NO SMOTE)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1 Score** | 0.6273 | Best balance of precision-recall |
| **Recall** | 0.6413 | Catches 64% of readmissions |
| **Precision** | 0.6139 | 61% of high-risk predictions are correct |
| **ROC AUC** | 0.7041 | Good discrimination ability |
| **Accuracy** | 0.6488 | Overall correctness |
| **Training Time** | 12.5s | Fast iteration |

**Risk Distribution:** Model produces realistic predictions ranging 9-35%, NOT clustering at extremes like earlier SMOTE-trained versions (which predicted 95%+ for all patients).

**Model Comparison (select runs from 490+ total):**

| Model | F1 | Recall | Precision | ROC-AUC | Notes |
|-------|-----|--------|-----------|---------|-------|
| **XGBoost** | 0.6273 | 0.6413 | 0.6139 | 0.7041 | Production model |
| Decision Tree | 0.6048 | 0.6345 | 0.5778 | 0.6636 | Interpretable baseline |
| Logistic Regression | ~0.58 | ~0.55 | ~0.61 | ~0.68 | Simple baseline |

---

## 🔧 Technical Architecture

```
Data (UCI) → Preprocessing → Feature Engineering (58 features)
     ↓
 MLflow Experiment Tracking (490+ runs)
     ↓
Model Training (9 algorithms × hyperparameter grids)
     ↓
Model Selection (XGBoost with scale_pos_weight)
     ↓
Model Registration (Databricks Unity Catalog)
     ↓
Deployment (Streamlit Cloud)
```

### Tech Stack
- **ML**: XGBoost, Scikit-learn
- **Experiment Tracking**: MLflow (Databricks)
- **Deployment**: Streamlit Cloud
- **Development**: Google Colab Pro (concurrent runtimes)
- **Version Control**: GitHub

---

## 📁 Repository Structure

```
diabetes-readmission-app/
│
├── app.py                      # Streamlit web application
├── model.pkl                   # Trained XGBoost model
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Local Deployment

```bash
# Clone repository
git clone https://github.com/yourusername/diabetes-readmission-app.git
cd diabetes-readmission-app

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Access at http://localhost:8501
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your forked repository
4. Deploy with `app.py` as main file

---

## 🔬 Full Reproduction (Training from Scratch)

### 1. Data Acquisition

**Dataset:** UCI Machine Learning Repository - Diabetes 130-US Hospitals (1999-2008)

```python
import pandas as pd

df = pd.read_csv('diabetic_data.csv')
print(f"Initial dataset: {df.shape}")  # (101,766 rows, 50 columns)
```

**Citation:**  
Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014. https://doi.org/10.1155/2014/781670

### 2. Data Preprocessing

**CRITICAL WARNING:** Do NOT use `df.dropna()` — it will reduce your dataset from 101,766 rows to 289 rows (99.7% loss). This was an actual bug in this project that cost 2 days of debugging.

**Correct approach:**

```python
# Drop only columns with >40% missing
missing_pct = df.isnull().sum() / len(df) * 100
cols_to_drop = missing_pct[missing_pct > 40].index  
# Drops: weight, payer_code, medical_specialty
df = df.drop(columns=cols_to_drop)

# Impute remaining missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)

print(f"After preprocessing: {df.shape}")  # (~70,000 rows, 45 columns)
```

**Target Variable Definition:**

```python
# IMPORTANT DECISION: Use "any readmission" not "<30 days"
# Reason: <30 day readmission has only 5% positive class → F1 scores 0.03-0.15 (unusable)
# "Any readmission" has 46% positive class → F1 scores 0.50-0.70 (viable)

df['readmitted_any'] = (df['readmitted'] != 'NO').astype(int)
print(df['readmitted_any'].value_counts(normalize=True))
# Expected: ~46% readmitted, ~54% not readmitted
```

### 3. Feature Engineering

Create 17 derived features from domain knowledge:

```python
# Medication complexity
df['total_medications'] = df[[med_cols]].apply(lambda row: sum(row != 'No'), axis=1)
df['medication_changes'] = df[[med_cols]].apply(lambda row: sum(row.isin(['Up', 'Down'])), axis=1)
df['insulin_prescribed'] = (df['insulin'] != 'No').astype(int)

# Utilization patterns
df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df['has_emergency_history'] = (df['number_emergency'] > 0).astype(int)
df['has_inpatient_history'] = (df['number_inpatient'] > 0).astype(int)

# Clinical complexity
df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures']
df['procedures_per_day'] = df['total_procedures'] / (df['time_in_hospital'] + 1)
df['multiple_diagnoses'] = (df['number_diagnoses'] > df['number_diagnoses'].median()).astype(int)

# Age encoding
age_map = {'[0-10)': 5, '[10-20)': 15, ..., '[90-100)': 95}
df['age_numeric'] = df['age'].map(age_map)
df['is_elderly'] = (df['age_numeric'] >= 70).astype(int)

# Risk interactions
df['elderly_emergency_risk'] = df['is_elderly'] * df['has_emergency_history']
df['complex_case'] = df['high_medication_load'] * df['multiple_diagnoses']
df['high_utilization'] = ((df['total_prior_visits'] > df['total_prior_visits'].quantile(0.75)) & 
                          (df['time_in_hospital'] > df['time_in_hospital'].median())).astype(int)

print(f"Total features: {df.shape[1]}")  # 58 features
```

**Categorical Encoding:**

```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3',
                    'metformin', 'insulin', 'change', 'diabetesMed', ...]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
```

### 4. Model Training with MLflow

**Setup:**

```python
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Configure MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/youremail@asu.edu/Diabetes_Production_Pipeline")

# Train-test split
X = df.drop(columns=['readmitted_any', 'readmitted', 'patient_nbr'])
y = df['readmitted_any']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Train All 9 Required Models:**

The project requirements mandated:
1. Classification Tree
2. Logistic Regression
3. Support Vector Machine
4. Neural Network
5. Naive Bayes
6. Random Forest
7. XGBoost
8. k-Nearest Neighbors
9. Ensemble

Each with 2-3 hyperparameters × 2+ values = 6-8 runs per model.

**Example: XGBoost Training**

```python
# CRITICAL: Use scale_pos_weight, NOT SMOTE
# Reason: SMOTE caused model to predict 95%+ risk for ALL patients (unusable)

hyperparameters = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.3},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1},
]

for params in hyperparameters:
    with mlflow.start_run(run_name=f"XGBoost_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"):
        
        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~1.17
        
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,  # This handles imbalance properly
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.sklearn.log_model(model, "model")
```

**Training Time Considerations:**

- Logistic Regression: 5-10 seconds
- Random Forest: 30-60 seconds
- XGBoost: 10-15 seconds
- **Neural Network: 2+ hours** (used early stopping; had 8 required configurations)
- **SVM: 2+ hours** (used 10-15% data subsampling; legitimate strategy for O(n²) algorithms)

**MLflow Logging Issue:**

Initial attempts used GridSearchCV with autologging, which only captured final best results (9 runs total). Professor required 20+ comparable runs. Solution:

```python
# Disable autologging
mlflow.sklearn.autolog(disable=True)

# Manually log every hyperparameter configuration
# Resulted in 490+ logged runs
```

Note: Required Colab runtime restart to clear persistent autolog state.

### 5. Model Selection and Registration

**Selection criteria:**
- Highest F1 score (balance of precision-recall)
- Realistic predictions (9-35% range, not 95%+ for everyone)
- Fast training time (enables iteration)

**Best model:** XGBoost (n_estimators=100, max_depth=5, learning_rate=0.3, scale_pos_weight=1.17)

**Register in Databricks:**

```python
from mlflow.models.signature import infer_signature

# Retrain best model
final_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.3,
    scale_pos_weight=1.17,
    random_state=42
)
final_model.fit(X_train, y_train)

# Create signature for deployment
signature = infer_signature(X_train, final_model.predict(X_train))
input_example = X_train.iloc[:5]

with mlflow.start_run(run_name="XGBoost_Production"):
    mlflow.sklearn.log_model(
        final_model,
        "model",
        signature=signature,
        input_example=input_example,
        registered_model_name="diabetes_readmission_production"
    )

# Export for Streamlit deployment
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
```

---

## 🚧 Project Challenges & Solutions

This section documents REAL challenges encountered, not sanitized success stories. Learning from mistakes is essential for ML practitioners.

### Challenge 1: Data Cleaning Disaster

**Problem:** Initial preprocessing code used `df.dropna()`, which drops any row containing ANY missing value. This reduced dataset from 101,766 rows to 289 rows (99.7% loss).

**Impact:** Lost 2 days debugging why model performance was terrible before discovering the data was essentially gone.

**Root Cause:** Blindly following "clean" data practices without validating data shapes after transformations.

**Solution:** Strategic missing data handling—dropped only 5 columns with >40% missing, used median/mode imputation for remaining.

**Lesson:** ALWAYS `print(df.shape)` after EVERY preprocessing step. Use `assert` statements in production.

### Challenge 2: Target Variable Redefinition

**Problem:** Initial target was "<30 day readmission" (clinical standard), which yielded only 5% positive class. Models achieved F1 scores of 0.03-0.15 (essentially random).

**Impact:** Week of hyperparameter tuning on a fundamentally unworkable problem.

**Root Cause:** Followed literature definition without considering ML feasibility.

**Solution:** Redefined target as "any readmission" → 46% positive class → F1 scores 0.50-0.70.

**Justification:** All readmissions are costly; predicting any readmission is still clinically valuable.

**Lesson:** Balance domain requirements with ML feasibility. Consult stakeholders about acceptable target redefinitions.

### Challenge 3: SMOTE Overfitting Crisis

**Problem:** Initial XGBoost model trained with SMOTE predicted >95% readmission risk for ALL patient profiles (young, healthy, elderly, sick—everyone flagged high-risk).

**Impact:** Model was clinically useless. Flagging everyone wastes hospital resources and provides no decision support.

**Root Cause:** Synthetic minority oversampling (SMOTE) created training data that didn't reflect reality. Model learned patterns from synthetic examples, not real patients.

**Solution:** Retrained WITHOUT SMOTE, using `scale_pos_weight=1.17` (actual class ratio) to handle imbalance.

**Result:** New model predicts realistic 9-35% risk range that varies appropriately by patient profile.

**Lesson:** SMOTE can cause overfitting. Always validate predictions against domain knowledge. For tree-based models, class weighting often works better than oversampling.

### Challenge 4: Training Time Issues

**Problem:** Neural Network (8 configurations) and SVM (8 configurations) each taking 2+ hours per run. Total: 32+ hours for just these two models, incompatible with project deadline.

**Impact:** Unable to complete all required models within timeline.

**Root Cause:** Dataset of 70K+ samples; SVM has O(n²) complexity; Neural Network requires many epochs.

**Solution:**
- Neural Network: Added early stopping (patience=10), reduced max_iter=100
- SVM: Subsampled to 10-15% of training data (~7-10K samples)

**Justification:** Subsampling for computationally expensive algorithms is legitimate practice in ML, especially during hyperparameter search.

**Result:** Training time reduced to 1-2 minutes per run while maintaining representative performance estimates.

**Lesson:** Simplify approaches under time constraints. 15% of data can still yield good hyperparameter rankings.

### Challenge 5: MLflow Logging Problems

**Problem:** Used GridSearchCV for hyperparameter tuning, but only final best result per model logged to MLflow. Showed only 9 runs total in Databricks, not the required 20+ comparable runs.

**Impact:** Couldn't demonstrate exhaustive model exploration for academic requirements.

**Root Cause:** GridSearchCV's internal cross-validation doesn't automatically log all iterations to MLflow.

**Solution:** 
- Disabled autologging (`mlflow.sklearn.autolog(disable=True)`)
- Manually logged every hyperparameter configuration as separate MLflow run
- Required Colab runtime restart to clear persistent autolog state from memory

**Result:** 490+ runs properly logged with consistent metrics (test_accuracy, test_f1, test_precision, test_recall, test_roc_auc).

**Lesson:** MLflow integration should be first implementation step, not a retrofit. Plan experiment naming conventions and metric standardization before training begins.

### Challenge 6: Model Registration Failures

**Problem:** Trained models couldn't be registered in Databricks Unity Catalog for deployment.

**Root Cause:** MLflow logging didn't include model signature (specification of input/output types and shapes).

**Solution:** Added signature inference and input examples:

```python
signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train.iloc[:5]

mlflow.sklearn.log_model(
    model, "model",
    signature=signature,
    input_example=input_example
)
```

**Lesson:** Production ML requires more than just model training—proper logging, signatures, and examples enable deployment.

### Challenge 7: Feature Mismatch in Deployment

**Problem:** Streamlit app crashed with "feature mismatch" errors. Model expected 58 features in specific order.

**Root Cause:** Web app's feature engineering pipeline didn't exactly match training preprocessing.

**Solution:** Documented exact preprocessing steps; created `prepare_input_data()` function that replicates training pipeline including:
- Feature calculation order
- Categorical encoding
- Derived feature creation

**Lesson:** The "model" is the entire pipeline (preprocessing + algorithm), not just the algorithm. Document everything.

---

## ⚖️ Ethical Considerations

### Demographic Bias

**Concern:** Training data (1999-2008, 130 US hospitals) may not represent current patient populations or underserved communities.

**Mitigation:** 
- Model performance should be evaluated separately by race, age, gender before clinical deployment
- Regular retraining on recent data required
- Not recommended for deployment without bias audit

### False Negative Risk

**Concern:** Model has 64% recall → misses 36% of readmissions. In healthcare, missing high-risk patients can have serious consequences.

**Mitigation:**
- Model is decision support tool, not replacement for clinical judgment
- Low-risk predictions should not override physician concerns
- System should be combined with existing protocols, not replace them

### Resource Allocation Fairness

**Concern:** Automated risk scoring could lead to unequal resource distribution if model exhibits demographic bias.

**Mitigation:**
- Transparent risk score thresholds
- Regular audits of intervention assignment by demographic groups
- Override mechanisms for care teams

### Algorithmic Transparency

**Concern:** XGBoost is less interpretable than logistic regression. Clinical staff may not understand why patient flagged high-risk.

**Future Work:**
- Add SHAP (SHapley Additive exPlanations) feature importance per prediction
- Provide plain-language explanations: "High risk due to: 3 prior hospitalizations, 18 medications, elderly"
- Build parallel interpretable model for validation

---

## 🎓 Academic Context

**Course:** CIS 508 - Machine Learning in Business  
**Institution:** W.P. Carey School of Business, Arizona State University  
**Project Type:** End-to-end ML pipeline with production deployment

**Assignment Requirements Met:**
- ✅ 9 classification algorithms implemented
- ✅ 2-3 hyperparameters per model × 2+ values each
- ✅ 490+ MLflow experiment runs logged
- ✅ Model registered in Databricks Unity Catalog  
- ✅ Production web application deployment
- ✅ Business problem formulation with ROI analysis
- ✅ Complete documentation of challenges and solutions

**Key Learning Outcomes:**
1. **Data quality > algorithm sophistication**: Preprocessing errors can destroy projects
2. **Target definition is critical**: Business problem formulation must balance domain requirements with ML feasibility
3. **Class imbalance requires careful handling**: SMOTE isn't always the answer; tree-based models often prefer weighting
4. **Experiment tracking from day 1**: Retrofitting MLflow is painful; plan upfront
5. **Domain validation is essential**: ML metrics alone don't guarantee useful predictions
6. **Production deployment ≠ just training**: Requires signatures, proper logging, feature pipeline documentation

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

1. **Temporal Validity**: Model trained on 1999-2008 data may not reflect current clinical practices, medications, or patient populations

2. **Calibration**: Probabilities may not reflect true readmission rates (e.g., 30% prediction ≠ exactly 30% chance). Calibration curve analysis needed.

3. **Feature Availability**: Requires 58 specific features at prediction time. Not all EHR systems capture these fields.

4. **Generalization**: Trained on 130 US hospitals; may not generalize to international settings or specific hospital types (rural, urban, academic, community)

5. **Class Imbalance**: Despite mitigation (scale_pos_weight), minority class slightly underrepresented. Could explore cost-sensitive learning.

6. **Interpretability**: XGBoost is black-box. Clinical adoption requires explainability (SHAP values, LIME).

### Planned Improvements

**Short-term (1-3 months):**
- Add SHAP feature importance visualization per prediction
- Implement probability calibration (Platt scaling or isotonic regression)
- Create automated bias audit report (performance by demographics)
- Build A/B testing framework for model versions

**Medium-term (3-6 months):**
- Retrain on more recent data (2015-2020) if available
- Implement temporal validation (train on older data, test on newer)
- Add real-time model monitoring (data drift detection)
- Integrate with EHR systems via FHIR API

**Long-term (6-12 months):**
- Build interpretable parallel model (e.g., rule list, attention mechanism)
- Expand to multi-class prediction (readmission reasons, not just binary)
- Develop causal inference models (treatment effect estimation)
- Clinical trial to measure actual impact on readmission rates

---

## 📚 References & Citations

### Dataset
Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014. https://doi.org/10.1155/2014/781670

### Technical Resources
- XGBoost Documentation: https://xgboost.readthedocs.io
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Streamlit Documentation: https://docs.streamlit.io

### Healthcare ML Best Practices
- Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.
- Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

---

## 🔐 License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](../../../../Downloads/LICENSE) file for details.

**What this means:**
- You can use, modify, and distribute this code
- Any derivative work must also be released under GPL v3
- You must disclose the source code of any modifications
- You cannot incorporate this into proprietary software

For full terms, see: https://www.gnu.org/licenses/gpl-3.0.en.html

---

## 👤 Author

**Chandra Carr**  
Graduate Student, W.P. Carey School of Business  
Arizona State University

**Contact:**
- Email: cgcarr@asu.edu
- GitHub: [@yourusername](https://github.com/WhiteRabbit-glitch)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/chandragcarr/)

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for diabetes dataset
- **Databricks Community Edition** for MLflow hosting
- **Streamlit** for free cloud deployment
- **Professor [Sang-Pil Han, Ph.D.], CIS 508** for project guidance
- **W.P. Carey School of Business** for computational resources
- **Jospeh Manning** for keeping me sane

---

## 📝 Project Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| Data Exploration | Week 1 | EDA, discovered data quality issues |
| Feature Engineering | Week 2 | 58 features, preprocessing pipeline |
| Model Training | Week 3 | 9 algorithms, 490+ MLflow runs |
| Model Selection | Week 4 | XGBoost optimization, SMOTE issue discovery |
| Deployment | Week 5 | Streamlit app, GitHub repository |
| Documentation | Week 6 | README, presentation, final report |

**Total Project Duration:** 6 weeks  
**Total MLflow Runs:** 490+  
**Final Deployment Date:** December 2024

---

## 🚨 Troubleshooting

### Model Loading Issues

**Problem:** Streamlit app fails to load model.pkl

**Solution:**
```python
# Verify pickle version compatibility
import pickle
print(pickle.format_version)

# Ensure model.pkl is in same directory as app.py
# Check file size (should be 1-5 MB for XGBoost)
```

### Feature Mismatch Errors

**Problem:** "Expected 58 features, got 45"

**Solution:** Verify feature engineering pipeline matches training exactly:
1. Check categorical encoding order
2. Verify all derived features calculated
3. Ensure same column order as training

### Unrealistic Predictions

**Problem:** All predictions >90% or all predictions <5%

**Solution:** This indicates SMOTE overfitting or improper scaling:
- Retrain without SMOTE, use scale_pos_weight instead
- Check that test data wasn't accidentally standardized with wrong scaler
- Verify y_proba uses predict_proba(), not predict()

### MLflow Autolog Issues

**Problem:** Metrics not logging or random run names like "carefree-hen-551"

**Solution:**
```python
# Disable autolog completely
mlflow.sklearn.autolog(disable=True)

# Restart Colab runtime to clear autolog state
# Then manually log all metrics
```

---

**⭐ If you find this project helpful, please star the repository!**

---

*Last updated: December 2025*
