# Student Performance Prediction System

A simple, complete AI/ML web application that predicts whether a student
will **Pass** or **Fail**, built entirely with local, open-source tools.

```
User → Frontend (HTML/CSS/JS) → Flask Backend → Locally Trained ML Model → Result → SQLite Database
```

No external AI API (OpenAI, Gemini, Claude, Hugging Face, or any cloud
prediction service) is used anywhere in this project. All predictions
happen inside Flask using a model trained locally with scikit-learn.

---

## 1. Project Idea

Predict whether a student will pass or fail their final exam, based on:
study hours, attendance (absences), past class failures, and marks from
two earlier assessment periods. This is a simple **binary classification**
problem — easy to explain in a viva/evaluation and reliable to implement
end-to-end.

## 2. Why This Satisfies the Challenge

- Uses only the mandated stack: HTML/CSS/JS, Flask, scikit-learn, pandas,
  NumPy, SQLite.
- The model is trained **once**, offline, with `train_model.py`, and saved
  to disk (`model/trained_model.pkl`). Flask **loads** this file — it never
  retrains on a request.
- No network calls happen at prediction time. Everything runs locally.
- Simple algorithm (Random Forest), a real public dataset, a clean 3-page
  UI, and a SQLite prediction history — matching every requirement in the
  brief.

## 3. Dataset

- **Name**: UCI "Student Performance" Data Set — Mathematics course file
  (`student-mat.csv`)
- **Source**: Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict
  Secondary School Student Performance*. UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/320/student+performance
- **Records**: 395 real students from two Portuguese secondary schools
- **Original columns**: 33 (demographic, social and school-related
  attributes, plus three period grades G1, G2, G3)
- **Missing values**: none

## 4. Input Features Used

| Feature     | Meaning                                   | Range         |
|-------------|--------------------------------------------|---------------|
| `studytime` | Weekly study time category                 | 1–4           |
| `absences`  | Number of school absences (attendance proxy)| 0–100        |
| `failures`  | Number of past class failures               | 0–4           |
| `G1`        | First-term grade (previous marks)           | 0–20          |
| `G2`        | Second-term grade (internal marks)          | 0–20          |

## 5. Target Variable

`pass` — engineered from the original `G3` (final grade, 0–20):

```
pass = 1  if G3 >= 10   (Pass)
pass = 0  if G3 < 10    (Fail)
```

This threshold (10/20) is the standard passing mark used in the original
dataset's grading scale and in published work on this dataset.

## 6. Data Preprocessing

1. Loaded the CSV with `pandas` (semicolon-delimited).
2. Checked for missing values (none found; a median-fill fallback exists
   in case any appear on a different run).
3. Selected the 5 input features above and engineered the `pass` target.
4. Scaled all features with `StandardScaler` (zero mean, unit variance) so
   the model isn't biased toward larger-magnitude columns like `absences`.

## 7. Train / Test Split

80% training / 20% testing, stratified by class to preserve the pass/fail
ratio (`train_test_split(..., test_size=0.2, stratify=y, random_state=42)`).

- Training samples: 316
- Testing samples: 79

## 8. Algorithm Selected

**Random Forest Classifier** (`n_estimators=200`, `max_depth=5`).

### Why this algorithm

- Handles a small mix of numeric features well without needing extensive
  tuning.
- More robust to overfitting than a single Decision Tree, while still
  being easy to explain ("many small decision trees vote on the answer").
- Naturally provides **feature importance**, useful for explaining which
  factors matter most.
- Avoids unnecessary complexity — no deep learning needed for a dataset
  this size.

## 9. Training Process

`train_model.py` performs, step by step:

1. Load dataset
2. Clean data / handle missing values
3. Select features and target
4. Scale features (`StandardScaler`)
5. Train/test split (80/20, stratified)
6. Train `RandomForestClassifier`
7. Evaluate on the held-out test set
8. Save the trained model (`model/trained_model.pkl`) and scaler
   (`model/scaler.pkl`) with `joblib`

## 10. Evaluation Metrics (test set)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8861 |
| Precision | 0.9583 |
| Recall    | 0.8679 |
| F1-score  | 0.9109 |

Confusion matrix:

```
                Predicted Fail   Predicted Pass
Actual Fail            24               2
Actual Pass             7              46
```

Feature importance (which inputs mattered most to the model):

```
G2 (internal marks)   : 60.2%
G1 (previous marks)   : 26.7%
failures              :  6.2%
absences              :  5.2%
studytime             :  1.8%
```

This matches intuition: recent grades (G1, G2) are the strongest
predictors of the final outcome, which is exactly what a real teacher
would expect.

## 11. How Prediction Works

1. User fills the form on the home page (study hours, absences, past
   failures, G1, G2).
2. Flask validates every field (required, numeric, within range).
3. The 5 values are assembled into a feature vector and scaled with the
   **same** `StandardScaler` fitted during training.
4. The locally saved `RandomForestClassifier` (`model/trained_model.pkl`)
   predicts Pass/Fail and a confidence score (`predict_proba`).
5. The input and prediction are saved to the SQLite `prediction` table.
6. The result page shows the prediction, confidence, and a short
   explanation.
7. Previous predictions can be browsed on the History page.

---

## 12. Project Structure

```
project/
│
├── app.py                 # Flask backend
├── train_model.py         # Offline model training script
├── requirements.txt
├── README.md
│
├── dataset/
│   └── student-mat.csv    # UCI Student Performance dataset (Math course)
│
├── model/
│   ├── trained_model.pkl  # Saved Random Forest model
│   ├── scaler.pkl         # Saved StandardScaler
│   └── metrics.txt        # Saved evaluation metrics
│
├── database/
│   └── app.db             # SQLite database (created at first run)
│
├── templates/
│   ├── index.html         # Home page / input form
│   ├── result.html        # Prediction result page
│   └── history.html       # Prediction history page
│
└── static/
    ├── style.css
    └── script.js
```

## 13. Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 14. How to Run

```bash
# Step 1: Train the model (only needs to be run once, or whenever you
# want to retrain)
python train_model.py

# Step 2: Start the Flask app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The SQLite database and its `prediction` table are created automatically
the first time the app starts.

## 15. How to Test the Application

1. **Home page** — open `/`, confirm the form and project description load.
2. **Valid prediction** — enter values such as study time = "5 to 10
   hours/week", absences = 2, failures = 0, G1 = 15, G2 = 16 → should
   predict **Pass** with high confidence.
3. **Weak-student prediction** — study time = "Less than 2 hours/week",
   absences = 40, failures = 3, G1 = 6, G2 = 5 → should predict **Fail**.
4. **Validation** — submit the form with a field left blank, or a value
   out of range (e.g. G1 = 99), and confirm a clear error message appears
   instead of a crash.
5. **History page** — open `/history` and confirm both test predictions
   above appear with the correct timestamp and result.

## 16. ML Explanation for Viva / Project Evaluation

- **Problem type**: Supervised binary classification.
- **Why classification, not regression**: The goal is a decision
  (Pass/Fail), which is more directly useful and easier to evaluate than
  predicting an exact numeric grade.
- **Why Random Forest**: Ensemble of decision trees; reduces overfitting
  compared to one tree, handles non-linear relationships between features,
  and needs minimal preprocessing/tuning — appropriate for a small,
  structured, tabular dataset like this one.
- **Why these 5 features**: They represent the information a school or
  student would realistically have *before* the final result — study
  habits, attendance, prior failures, and marks from earlier assessments
  — mirroring how a teacher would informally judge risk.
- **How the model was validated**: An 80/20 train/test split was used, and
  performance was measured with accuracy, precision, recall, F1-score and
  a confusion matrix, so both false positives (predicting Pass when the
  student actually fails) and false negatives are visible.
- **How the app avoids retraining on each request**: The model is trained
  once by `train_model.py` and saved with `joblib`; `app.py` loads that
  file once at startup and reuses it for every prediction.

## 17. Limitations

- Trained on only 395 records from two Portuguese schools — may not
  generalize to other regions, grading systems, or age groups.
- Only 5 of the original 30+ attributes are used, for simplicity; other
  factors (family background, health, extracurriculars) are not
  considered.
- The Pass/Fail threshold (G3 ≥ 10) is a simplification of real academic
  outcomes, which are often more nuanced (grades, honors, etc.).
- Model performance (accuracy ~89%) is good for a class demo but not
  suitable for high-stakes, real-world academic decisions.

## 18. Possible Future Improvements

- Add more input features (parental education, internet access, health,
  social activity) for richer predictions.
- Predict a performance category (e.g. Excellent / Good / Average / Poor)
  instead of just Pass/Fail.
- Add basic authentication so each teacher/student sees only their own
  history.
- Add a chart on the history page (e.g. pass-rate over time) using
  matplotlib or a lightweight JS charting library.
- Package the SQLite history as a CSV export.
- Cross-validate and compare multiple simple models (Logistic Regression,
  Decision Tree, Random Forest) and report the best one automatically.

---

**Technology stack**: Python · Flask · HTML5 · CSS3 · JavaScript ·
scikit-learn · pandas · NumPy · SQLite · joblib
