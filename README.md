# Salary AI Engine

AI-powered Employee Salary Prediction web application built with Flask and Scikit-learn.

## Features

- Login system (Username + Password)
- Predict salary based on:
  - Years of Experience
  - Performance Score
  - Certifications Count
  - Weekly Work Hours
  - Education Level (Bachelor's / Master's / PhD)
  - Job Role (HR, Software Engineer, Analyst, Manager, Data Scientist)
- Dark / Light theme toggle
- Model persistence using joblib
- Responsive modern UI

## Login Credentials

| Username | Password   |
|----------|------------|
| admin    | admin123   |
| user     | user123    |

## How to Run Locally

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
