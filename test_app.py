from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_predict_valid():
    response = client.post("/predict", json={
        "VendorID": 1,
        "passenger_count": 2,
        "trip_distance": 5.1,
        "RatecodeID": 1,
        "PULocationID": 100,
        "DOLocationID": 200,
        "payment_type": 1,
        "fare_amount": 35,
        "extra": 0,
        "mta_tax": 0.5,
        "tolls_amount": 0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0,
        "trip_duration_minutes": 15,
        "trip_speed_mph": 20,
        "pickup_hour": 12,
        "pickup_day_of_week": 3,
        "log_trip_distance": 1.6,
        "fare_per_mile": 6.8,
        "fare_per_minute": 2.3,
        "tpep_pickup_datetime_hour": 12,
        "tpep_pickup_datetime_day": 10,
        "tpep_pickup_datetime_month": 5,
        "tpep_dropoff_datetime_hour": 12,
        "tpep_dropoff_datetime_day": 10,
        "tpep_dropoff_datetime_month": 5,
        "store_and_fwd_flag_Y": False,

        "pickup_Borough_Brooklyn": False,
        "pickup_Borough_EWR": False,
        "pickup_Borough_Manhattan": True,
        "pickup_Borough_N_A": False,
        "pickup_Borough_Queens": False,
        "pickup_Borough_Staten_Island": False,
        "pickup_Borough_Unknown": False,

        "dropoff_Borough_Brooklyn": False,
        "dropoff_Borough_EWR": False,
        "dropoff_Borough_Manhattan": True,
        "dropoff_Borough_N_A": False,
        "dropoff_Borough_Queens": False,
        "dropoff_Borough_Staten_Island": False,
        "dropoff_Borough_Unknown": False
    })

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_id" in data

# --- Validation tests ---
def test_predict_missing_field():
    """Missing required field should return 422."""
    response = client.post("/predict", json={
        "VendorID": 1,
        "passenger_count": 2,
    # missing other fields
    })
    assert response.status_code == 422


def test_predict_invalid_type():
    response = client.post("/predict", json={
        "VendorID": "wrong",  # invalid
        "passenger_count": 2,
        "trip_distance": 5.1,
        "RatecodeID": 1,
        "PULocationID": 100,
        "DOLocationID": 200,
        "payment_type": 1,
        "fare_amount": 35,
        "extra": 0,
        "mta_tax": 0.5,
        "tolls_amount": 0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0,
        "trip_duration_minutes": 15,
        "trip_speed_mph": 20,
        "pickup_hour": 12,
        "pickup_day_of_week": 3,
        "log_trip_distance": 1.6,
        "fare_per_mile": 6.8,
        "fare_per_minute": 2.3,
        "tpep_pickup_datetime_hour": 12,
        "tpep_pickup_datetime_day": 10,
        "tpep_pickup_datetime_month": 5,
        "tpep_dropoff_datetime_hour": 12,
        "tpep_dropoff_datetime_day": 10,
        "tpep_dropoff_datetime_month": 5,
        "store_and_fwd_flag_Y": False,

        "pickup_Borough_Brooklyn": False,
        "pickup_Borough_EWR": False,
        "pickup_Borough_Manhattan": True,
        "pickup_Borough_N_A": False,
        "pickup_Borough_Queens": False,
        "pickup_Borough_Staten_Island": False,
        "pickup_Borough_Unknown": False,

        "dropoff_Borough_Brooklyn": False,
        "dropoff_Borough_EWR": False,
        "dropoff_Borough_Manhattan": True,
        "dropoff_Borough_N_A": False,
        "dropoff_Borough_Queens": False,
        "dropoff_Borough_Staten_Island": False,
        "dropoff_Borough_Unknown": False
    })

    assert response.status_code == 422

def test_predict_out_of_range():
    """Negative value when gt=0 should return 422."""
    response = client.post("/predict", json={
    "distance":-1,
    "hour": 12,
    "passenger_count": 2,
    "fare": 35,
    })
    assert response.status_code == 422


# --- Edge case tests ---
def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "metrics" in data

def test_batch_prediction():
    record = {
        "VendorID": 1,
        "passenger_count": 2,
        "trip_distance": 5.1,
        "RatecodeID": 1,
        "PULocationID": 100,
        "DOLocationID": 200,
        "payment_type": 1,
        "fare_amount": 35,
        "extra": 0,
        "mta_tax": 0.5,
        "tolls_amount": 0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0,
        "trip_duration_minutes": 15,
        "trip_speed_mph": 20,
        "pickup_hour": 12,
        "pickup_day_of_week": 3,
        "log_trip_distance": 1.6,
        "fare_per_mile": 6.8,
        "fare_per_minute": 2.3,
        "tpep_pickup_datetime_hour": 12,
        "tpep_pickup_datetime_day": 10,
        "tpep_pickup_datetime_month": 5,
        "tpep_dropoff_datetime_hour": 12,
        "tpep_dropoff_datetime_day": 10,
        "tpep_dropoff_datetime_month": 5,
        "store_and_fwd_flag_Y": False,

        "pickup_Borough_Brooklyn": False,
        "pickup_Borough_EWR": False,
        "pickup_Borough_Manhattan": True,
        "pickup_Borough_N_A": False,
        "pickup_Borough_Queens": False,
        "pickup_Borough_Staten_Island": False,
        "pickup_Borough_Unknown": False,

        "dropoff_Borough_Brooklyn": False,
        "dropoff_Borough_EWR": False,
        "dropoff_Borough_Manhattan": True,
        "dropoff_Borough_N_A": False,
        "dropoff_Borough_Queens": False,
        "dropoff_Borough_Staten_Island": False,
        "dropoff_Borough_Unknown": False
    }

    response = client.post("/predict/batch", json={
        "records": [record, record, record]
    })

    assert response.status_code == 200
    assert response.json()["count"] == 3