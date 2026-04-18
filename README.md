Here is a concise, professional **README.md** tailored for your Assignment 4 submission.

---

# NYC Taxi Tip Prediction Service

A containerized machine learning pipeline and REST API for predicting taxi tip amounts using the NYC Yellow Taxi dataset.

##  Overview
- **ML Pipeline:** Automated data cleaning and feature engineering using Polars and Pandas.
- **Model:** Random Forest Regressor trained on 42 specific features.
- **API:** FastAPI service for real-time inference.
- **Tracking:** MLflow integration for experiment logging and model versioning.
- **Orchestration:** Docker Compose for multi-container deployment.

##  Repository Structure
- `app.py` – FastAPI application logic.
- `3610a4.ipynb` – Pipeline orchestration and testing.
- `Dockerfile` – API container configuration.
- `docker-compose.yml` – Service definitions for API and MLflow.
- `requirements.txt` – Python dependencies.

## 🛠️ Setup & Usage

### 1. Prerequisites
Ensure **Docker Desktop** is running and you are in the project root.

### 2. Start Services
```bash
docker compose up --build -d
```
* **API:** `http://localhost:8000/docs`
* **MLflow UI:** `http://localhost:5000`

### 3. Testing Predictions
Run the loop inside `Assignment_Notebook.ipynb` or send a POST request to `http://localhost:8000/predict` with the 42-field JSON payload.

### 4. Shutdown
```bash
docker compose down
```

## Features
The model utilizes 42 engineered features, including:
- **Temporal:** Pickup/Dropoff hour, day, month, and weekday.
- **Geospatial:** Pickup and Dropoff Boroughs (One-Hot Encoded).
- **Calculated:** Trip duration, speed (mph), fare per mile, and fare per minute.
