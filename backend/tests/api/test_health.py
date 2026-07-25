from fastapi.testclient import TestClient

from daikonstudio.interface.app import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_service_name():
    client = TestClient(create_app())
    assert response_json(client)["service"] == "daikon-studio"


def response_json(client):
    return client.get("/version").json()
