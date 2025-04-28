import pytest
from fastapi.testclient import TestClient
from src.api.app import app 

client = TestClient(app)

def test_classify_text():
    response = client.post(
        "/classify/text",
        json={"text": "This is a test news article.", "explanation": True},
    )
    assert response.status_code == 200
    assert "classification" in response.json()
    assert "explanation" in response.json()

def test_classify_text_no_explanation():
    response = client.post(
        "/classify/text",
        json={"text": "Another test article."},
    )
    assert response.status_code == 200
    assert "classification" in response.json()
    assert "explanation" not in response.json()

def test_classify_text_empty():
    response = client.post(
        "/classify/text",
        json={"text": ""},
    )
    assert response.status_code == 400
    assert "error" in response.json()