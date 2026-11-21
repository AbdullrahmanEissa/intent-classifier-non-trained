# Intent Classification with DVC & MLflow

An end-to-end Intent Classification pipeline using Scikit-Learn, DVC (with AWS S3), MLflow (with SQLite backend), and a Flask inference API.

## Project Structure

```text
.
├── dataset.csv
├── dataset.csv.dvc
├── train.py
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

## 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. DVC Data Versioning (S3 Bucket: `intent-training`)

### Configure AWS Credentials
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Pull Existing Data from S3
```bash
dvc pull
```

### Track & Push Updated Data or Models to S3
```bash
dvc add dataset.csv intent_model.pkl
git add dataset.csv.dvc intent_model.pkl.dvc .gitignore
git commit -m "Update dataset and model artifacts"
dvc push
git push
```

## 3. Start MLflow Tracking Server (SQLite)

Start the MLflow UI and tracking server on port `5000` using a local SQLite database (`mlflow.db`):

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

## 4. Train Model & View Experiments

In a new terminal (with your virtual environment activated), run training:

```bash
python train.py
```

Open your browser to **`http://127.0.0.1:5000`** to view:
- **Experiments (`intent-classification`):** Compare runs, hyperparameters (`max_iter`, n-gram ranges), and `train_accuracy`.
- **Models (`intent-classifier`):** Inspect registered model versions and artifacts.

## 5. Run the Inference API & Web UI

Start the Flask app on port `8000`:

```bash
python app.py
```

- **Web UI:** Open `http://127.0.0.1:8000`
- **API Endpoint:**
  ```bash
  curl -X POST http://127.0.0 \
    -H "Content-Type: application/json" \
    -d '{"text": "cancel subscription"}'
  ```
