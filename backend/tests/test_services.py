import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_list_exercises(client):
    response = await client.get("/api/exercises?language=es")
    assert response.status_code == 200
    data = response.json()
    assert "exercises" in data
    assert len(data["exercises"]) > 0


@pytest.mark.anyio
async def test_chat_stub(client):
    response = await client.post(
        "/api/tutor/chat",
        json={"message": "Hola", "language": "es", "history": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
