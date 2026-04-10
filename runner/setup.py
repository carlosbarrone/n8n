import os
from typing import Any

from psycopg import connect
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "experience")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "n8n")
POSTGRES_USER = os.getenv("POSTGRES_USER", "n8n")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
SETUP_KEY = os.getenv("RUNNER_SETUP_KEY", "runner_setup_v1")


def pg_dsn() -> str:
    return (
        f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} "
        f"user={POSTGRES_USER} password={POSTGRES_PASSWORD}"
    )


def check_postgres() -> None:
    with connect(pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user;")
            db_name, db_user = cur.fetchone()
            print(f"PostgreSQL connected: db={db_name}, user={db_user}")


def ensure_setup_table(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS setup_state (
                setup_key TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
    conn.commit()


def is_setup_marked(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM setup_state WHERE setup_key = %s", (SETUP_KEY,))
        return cur.fetchone() is not None


def mark_setup_complete(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO setup_state (setup_key)
            VALUES (%s)
            ON CONFLICT (setup_key) DO NOTHING
            """,
            (SETUP_KEY,),
        )
    conn.commit()


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Qdrant collection created: {COLLECTION_NAME}")
    else:
        print(f"Qdrant collection already exists: {COLLECTION_NAME}")


def collection_exists(client: QdrantClient) -> bool:
    existing = {c.name for c in client.get_collections().collections}
    return COLLECTION_NAME in existing


def upsert_point(client: QdrantClient, point_id: int, vector: list[float], payload: dict[str, Any]) -> None:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{"id": point_id, "vector": vector, "payload": payload}],
    )


def search(client: QdrantClient, query_vector: list[float], limit: int = 5) -> list[Any]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )
    return result.points


if __name__ == "__main__":
    check_postgres()
    qdrant = get_qdrant_client()

    with connect(pg_dsn()) as conn:
        ensure_setup_table(conn)
        already_marked = is_setup_marked(conn)
        already_has_collection = collection_exists(qdrant)

        if already_marked and already_has_collection:
            print("Setup already completed; skipping initialization")
        else:
            ensure_collection(qdrant)

            example_vector = [0.001] * VECTOR_SIZE
            upsert_point(
                qdrant,
                point_id=1,
                vector=example_vector,
                payload={"source": "runner_setup"},
            )
            results = search(qdrant, query_vector=example_vector, limit=1)
            print(f"Qdrant search ok, hits={len(results)}")

            mark_setup_complete(conn)
            print("Setup completed and marked")