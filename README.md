# Salary AI Engine

Flask web app that predicts employee salary using Random Forest with numeric + categorical features.

## Features
- Years of Experience, Performance Score, Certifications, Work Hours
- Education Level (Bachelor's / Master's / PhD)
- Job Role (HR, Software Engineer, Analyst, Manager, Data Scientist)
- Dark / Light theme
- Model persistence with joblib

## How to Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy
Ready for Render / Railway / Heroku (Procfile included).
