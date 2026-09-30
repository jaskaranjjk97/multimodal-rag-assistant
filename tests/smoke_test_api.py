import requests


BASE_URL = "http://localhost:8000"


def run_health_check():
    response = requests.get(f"{BASE_URL}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "rag-assistant"


def run_rag_query():
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "question": "What was Acme Technologies revenue in 2024?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data

    assert "10 crore" in data["answer"].lower()

    assert len(data["citations"]) > 0


if __name__ == "__main__":
    run_health_check()
    run_rag_query()

    print("Docker smoke tests passed.")