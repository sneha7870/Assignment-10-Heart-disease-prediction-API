# Heart Disease Risk Prediction — End-to-End ML Deployment

## Objective
Build a machine learning model that predicts whether a patient is at risk of heart disease
based on clinical parameters, expose it as a REST API using Flask, and deploy it as a live
web service on Render — for a healthcare organization automating patient risk screening.

## Dataset Link
[Heart Disease Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

`heart.csv` (303 records, 13 clinical features + target) is included in this repository per
the assignment's repository structure requirements.

## Libraries Used
- `pandas` — data loading and handling
- `scikit-learn` — `train_test_split`, `RandomForestClassifier`, `accuracy_score`,
  `classification_report`
- `joblib` — model serialization
- `flask` — REST API
- `gunicorn` — production WSGI server (used by Render)

## Methodology
1. **Data Understanding & Preprocessing** (`train_model.py`) — Loaded `heart.csv` with
   Pandas, displayed the first five records, identified the 13 numerical clinical features
   and the `target` variable (0 = no disease, 1 = heart disease), checked for missing values
   (none found), and split the data 80/20 (stratified) into train/test sets.
2. **Model Development** — Trained a `RandomForestClassifier` (200 trees) on the training
   set, evaluated it on the test set, and serialized the trained model to `model.pkl` with
   `joblib`.
3. **API Development** (`app.py`) — Built a Flask REST API that loads `model.pkl` at
   startup, accepts patient clinical details as JSON on `POST /predict`, and returns a JSON
   prediction (plus the model's predicted probability). A `GET /health` endpoint is included
   for uptime checks, and `GET /` renders a simple status/usage page.
4. **Deployment** — Pushed the project to a public GitHub repository and deployed it on
   Render as a live web service using Gunicorn.

## Model Architecture / Approach
- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `random_state=42`)
- **Features (13):** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`,
  `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target:** `target` (0 = No Disease, 1 = Heart Disease)
- **Train/test split:** 80% / 20%, stratified

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 0.8197 |

**Classification report (test set, 61 samples):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No Disease | 0.95 | 0.64 | 0.77 |
| Heart Disease | 0.76 | 0.97 | 0.85 |

## API Usage

**POST** `/predict`

Request body (all 13 fields required):
```json
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

Response:
```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.655
}
```

**GET** `/health` → `{"status": "ok"}`

## Running Locally
```bash
pip install -r requirements.txt
python train_model.py      # trains the model and writes model.pkl
python app.py               # starts the Flask API on http://localhost:5000
```

## Deploying on Render
1. Push this repository to GitHub (public).
2. On [Render](https://render.com), create a **New Web Service** and connect the GitHub repo.
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Deploy. Render will assign a public URL like `https://<your-service-name>.onrender.com`.
5. Test it: `POST https://<your-service-name>.onrender.com/predict` with the JSON body above.

**Deployed Application URL:** _add your live Render URL here after deployment_

## Conclusion
The Random Forest model reached 82% test accuracy in predicting heart disease risk from 13
clinical parameters, with notably high recall (97%) for the Heart Disease class — important
in a screening context, where missing an at-risk patient is more costly than a false alarm.
The main challenges in deployment were making sure the exact library versions used to train
and pickle the model (`scikit-learn`, `pandas`) matched the versions installed on the Render
server, since a mismatch can silently break `joblib.load()`, and ensuring the Flask app binds
to the `PORT` environment variable Render provides rather than a hardcoded port. This project
highlights why MLOps practices matter: a model is only useful once it's reliably packaged,
version-controlled, and served through an API that other systems (or a hospital's intake
software) can call in real time, rather than living only inside a notebook.

## Repository Structure
```
HeartDiseaseDeployment/
│
├── app.py                # Flask REST API
├── model.pkl              # trained model (Random Forest)
├── requirements.txt
├── README.md
├── train_model.py         # data loading, training, evaluation, serialization
├── heart.csv              # dataset
└── templates/
    └── index.html         # simple status page
```
