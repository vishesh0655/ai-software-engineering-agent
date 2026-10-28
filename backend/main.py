"""Start the API using API_HOST and API_PORT from the environment/.env."""
import uvicorn

from backend.settings import settings


if __name__ == "__main__":
    uvicorn.run("backend.api:app", host=settings.api_host, port=settings.api_port)
