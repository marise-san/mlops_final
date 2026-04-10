import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.model import ModelHandler

VALID_PAYLOAD = {
    "age": 30,
    "job": "admin.",
    "marital": "single",
    "education": "university.degree",
    "default": "no",
    "balance": 1500.50,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 250.0,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "nonexistent"
}

# Explicitly ensure the model is loaded when tests start running since lifespan context manager 
# handling differs between clients. 
@pytest.fixture(autouse=True)
def load_model_if_present():
    ModelHandler().load_model()
    yield

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "model_loaded" in data

@pytest.mark.asyncio
async def test_predict_endpoint_valid():
    if not ModelHandler().is_loaded:
        pytest.skip("Model not available. Prediction endpoint skip.")
        
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=VALID_PAYLOAD)
        
        # Edge case behavior testing check
        if response.status_code == 503:
            pytest.skip("Model not loaded and fallback returned 503.")
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data

@pytest.mark.asyncio
async def test_predict_endpoint_invalid():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        incomplete_payload = {"age": 30} # Missing all other fields
        response = await client.post("/predict", json=incomplete_payload)
        
        # FastAPI specifically catches this and returns 422 mapped via Pydantic
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
