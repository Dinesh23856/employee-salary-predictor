import os
import joblib
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

model_pipeline = None

NUMERIC_FEATURES = ['years_experience', 'performance_score', 'certifications_count', 'work_hours_per_week']
CATEGORICAL_FEATURES = ['education_level', 'job_role']
TARGET_COL = 'salary'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'employee_salary_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'salary_model.joblib')

FEATURE_RANGES = {
    'years_experience': (0.0, 40.0),
    'performance_score': (1.0, 5.0),
    'certifications_count': (0, 10),
    'work_hours_per_week': (10.0, 80.0),
}

ALLOWED_EDUCATION = ["Bachelor's", "Master's", "PhD"]
ALLOWED_JOB_ROLES = ['HR', 'Software Engineer', 'Analyst', 'Manager', 'Data Scientist']


def train_model_from_csv():
    global model_pipeline
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"'{CSV_PATH}' not found.")

    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1))
    ])

    print("Training Random Forest on numeric + categorical features...")
    model_pipeline.fit(X, y)
    print("Model training complete!")
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_or_train_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
            return
        except Exception as e:
            print(f"Failed to load model ({e}). Retraining...")
    print("No valid model found. Training new one...")
    train_model_from_csv()


def validate_input(data: dict) -> tuple:
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for key in required:
        if key not in data:
            return False, f"Missing required field: {key}"
    try:
        years = float(data['years_experience'])
        perf = float(data['performance_score'])
        certs = int(data['certifications_count'])
        hours = float(data['work_hours_per_week'])
    except (TypeError, ValueError):
        return False, "Numeric fields must be valid numbers."

    checks = [
        (FEATURE_RANGES['years_experience'][0] <= years <= FEATURE_RANGES['years_experience'][1],
         "years_experience must be between 0 and 40"),
        (FEATURE_RANGES['performance_score'][0] <= perf <= FEATURE_RANGES['performance_score'][1],
         "performance_score must be between 1.0 and 5.0"),
        (FEATURE_RANGES['certifications_count'][0] <= certs <= FEATURE_RANGES['certifications_count'][1],
         "certifications_count must be between 0 and 10"),
        (FEATURE_RANGES['work_hours_per_week'][0] <= hours <= FEATURE_RANGES['work_hours_per_week'][1],
         "work_hours_per_week must be between 10 and 80"),
        (data['education_level'] in ALLOWED_EDUCATION,
         f"education_level must be one of: {', '.join(ALLOWED_EDUCATION)}"),
        (data['job_role'] in ALLOWED_JOB_ROLES,
         f"job_role must be one of: {', '.join(ALLOWED_JOB_ROLES)}"),
    ]
    for ok, msg in checks:
        if not ok:
            return False, msg
    return True, ""


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Salary Predictor Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = { darkMode: 'class', theme: { extend: { colors: { brand: { 500: '#6366f1', 600: '#4f46e5' } } } } }
    </script>
    <style>
        body { background-size: 400% 400%; animation: gradientBg 15s ease infinite; }
        .light-bg { background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab); background-size: 400% 400%; }
        .dark-bg { background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #090d16); background-size: 400% 400%; }
        @keyframes gradientBg { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.5; animation: float 10s infinite alternate ease-in-out; pointer-events: none; }
        @keyframes float { 0% { transform: translate(0, 0) scale(1); } 100% { transform: translate(60px, 80px) scale(1.2); } }
        .glass { background: rgba(255,255,255,0.45); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.4); }
        .dark .glass { background: rgba(15,23,42,0.65); border: 1px solid rgba(255,255,255,0.1); }
        .btn-active:active { transform: scale(0.96); }
        select { appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239ca3af'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 0.75rem center; background-size: 1.25rem; }
    </style>
</head>
<body id="bgBody" class="dark-bg text-gray-800 dark:text-gray-100 min-h-screen transition-colors duration-500 flex flex-col justify-center items-center p-4 relative overflow-x-hidden">
    <div class="orb w-72 h-72 bg-indigo-500 top-10 left-10"></div>
    <div class="orb w-96 h-96 bg-purple-500 bottom-10 right-10" style="animation-delay: -5s;"></div>

    <div class="w-full max-w-2xl flex justify-between items-center mb-6 z-10">
        <div>
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">Salary AI Engine</h1>
            <p class="text-sm text-gray-300 dark:text-gray-400">Advanced ML Compensation Estimator</p>
        </div>
        <button id="themeToggle" class="p-3 rounded-full glass hover:bg-white/20 dark:hover:bg-black/30 transition-all btn-active">
            <span id="themeIcon" class="text-xl">☀️</span>
        </button>
    </div>

    <div class="w-full max-w-2xl glass rounded-3xl p-8 shadow-2xl z-10">
        <form id="predictionForm" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-sm font-medium mb-2">Years of Experience</label>
                    <input type="number" step="0.1" name="years_experience" min="0" max="40" value="3.5" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Performance Score (1.0 - 5.0)</label>
                    <input type="number" step="0.1" name="performance_score" min="1.0" max="5.0" value="4.2" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Certifications Count</label>
                    <input type="number" name="certifications_count" min="0" max="10" value="2" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Weekly Work Hours</label>
                    <input type="number" step="0.1" name="work_hours_per_week" min="10" max="80" value="40" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Education Level</label>
                    <select name="education_level" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="Bachelor's">Bachelor's</option>
                        <option value="Master's" selected>Master's</option>
                        <option value="PhD">PhD</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Job Role</label>
                    <select name="job_role" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="HR">HR</option>
                        <option value="Software Engineer">Software Engineer</option>
                        <option value="Analyst">Analyst</option>
                        <option value="Manager">Manager</option>
                        <option value="Data Scientist" selected>Data Scientist</option>
                    </select>
                </div>
            </div>
            <button type="submit" id="predictBtn"
                class="w-full py-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 text-white font-bold rounded-xl shadow-lg transition-all btn-active">
                Predict Salary
            </button>
        </form>
        <div id="resultContainer" class="hidden mt-8 p-6 rounded-2xl bg-indigo-500/10 border border-indigo-400/30 text-center">
            <span class="text-sm font-semibold uppercase text-indigo-300 tracking-wider">Estimated Annual Salary</span>
            <div id="predictedSalary" class="text-4xl font-extrabold text-indigo-400 mt-2">₹0</div>
        </div>
    </div>

    <script>
        document.getElementById('themeToggle').addEventListener('click', () => {
            const html = document.documentElement;
            const body = document.getElementById('bgBody');
            const icon = document.getElementById('themeIcon');
            if (html.classList.contains('dark')) {
                html.classList.remove('dark'); body.classList.remove('dark-bg'); body.classList.add('light-bg'); icon.textContent = '🌙';
            } else {
                html.classList.add('dark'); body.classList.remove('light-bg'); body.classList.add('dark-bg'); icon.textContent = '☀️';
            }
        });

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = Object.fromEntries(new FormData(e.target).entries());
            const btn = document.getElementById('predictBtn');
            btn.disabled = true; btn.textContent = 'Predicting...';
            try {
                const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const result = await res.json();
                if (result.status === 'success') {
                    document.getElementById('predictedSalary').textContent = '₹' + Number(result.prediction).toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
                    document.getElementById('resultContainer').classList.remove('hidden');
                } else {
                    alert(result.message || 'Error');
                }
            } catch (err) { alert('Network error'); }
            finally { btn.disabled = false; btn.textContent = 'Predict Salary'; }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    status = 'ok' if model_pipeline is not None else 'model_not_loaded'
    return jsonify({'status': status, 'model_loaded': model_pipeline is not None}), 200 if model_pipeline is not None else 503


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400

        input_df = pd.DataFrame([{
            'years_experience': float(data['years_experience']),
            'performance_score': float(data['performance_score']),
            'certifications_count': int(data['certifications_count']),
            'work_hours_per_week': float(data['work_hours_per_week']),
            'education_level': data['education_level'],
            'job_role': data['job_role']
        }])
        prediction = model_pipeline.predict(input_df)[0]
        return jsonify({'status': 'success', 'prediction': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


load_or_train_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
