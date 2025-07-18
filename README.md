# AI-Powered Lead Scoring System for CRMs

## Overview

This project is an internship work focused on building an AI-powered lead scoring system for CRM platforms. It uses machine learning (XGBoost) to predict the likelihood of a lead converting, based on historical CRM data. The solution includes model training, feature engineering, and a Streamlit web app for real-time lead scoring.

---

## Features

- **Automated Lead Scoring:** Predicts conversion probability for CRM leads.
- **Model Training & Tuning:** Uses XGBoost with hyperparameter optimization.
- **Feature Engineering:** Handles categorical and numerical CRM features.
- **Streamlit Web App:** User-friendly UI for real-time predictions.
- **Model & Encoder Persistence:** Saves trained model and label encoders for deployment.

---

## How It Works

1. **Data Preparation:**  
   - Cleaned CRM lead data is used (`data/cleaned_lead_scoring.csv`).
   - Categorical features are label-encoded.

2. **Model Training:**  
   - XGBoost classifier is trained and tuned using GridSearchCV.
   - Model and encoders are saved as `xgboost_lead_model.pkl` and `label_encoders.pkl`.

3. **Web App:**  
   - Streamlit app (`app.py`) loads the model and encoders.
   - Users input lead details; the app predicts conversion and shows probability.

---

## Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone <repo-url>
   cd AI-Powered-Lead-Scoring-System-for-CRMs
   ```

2. **Install Requirements**
   ```sh
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**
   - Run `algo_finetuning.py` to train and save the model and encoders.

4. **Run the Streamlit App**
   ```sh
   streamlit run app.py
   ```

---

## File Structure

- `data/cleaned_lead_scoring.csv` — Cleaned CRM leads data
- `algo_finetuning.py` — Model training and tuning script
- `app.py` — Streamlit web app for lead scoring
- `xgboost_lead_model.pkl` — Saved trained model
- `label_encoders.pkl` — Saved label encoders
- `requirements.txt` — Python dependencies

---

## Usage

- Open the Streamlit app.
- Enter lead details in the form.
- Click "Predict Lead Conversion" to get the conversion prediction and probability.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

---

## Author

Sivakumar Balaji

AI-Powered Lead Scoring System