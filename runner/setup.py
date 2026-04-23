import os
import time
from pathlib import Path
from typing import Any

from psycopg import connect
from psycopg import OperationalError
from psycopg.sql import SQL, Identifier
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION_NAMES = os.getenv("QDRANT_COLLECTION_NAMES").split(",")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "job_tracker")
SETUP_KEY = os.getenv("RUNNER_SETUP_KEY", "default_setup_key")
SQL_DIR = "/app/scripts/sql"


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


def wait_for_postgres(max_attempts: int = 30, delay_seconds: int = 2) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            check_postgres()
            return
        except OperationalError as error:
            if attempt == max_attempts:
                raise
            print(
                f"PostgreSQL not ready yet (attempt {attempt}/{max_attempts}): {error}. "
                f"Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    for collection_name in QDRANT_COLLECTION_NAMES:
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"Qdrant collection created: {collection_name}")
        else:
            print(f"Qdrant collection already exists: {collection_name}")


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    existing = {c.name for c in client.get_collections().collections}
    return collection_name in existing


def upsert_point(client: QdrantClient, collection_name: str, point_id: int, vector: list[float], payload: dict[str, Any]) -> None:
    client.upsert(
        collection_name=collection_name,
        points=[{"id": point_id, "vector": vector, "payload": payload}],
    )


def search(client: QdrantClient, collection_name: str, query_vector: list[float], limit: int = 5) -> list[Any]:
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
    )
    return result.points


def ensure_schema(conn: Any, schema_name: str) -> None:
    print(f"Ensuring PostgreSQL schema exists: {schema_name}")
    with conn.cursor() as cur:
        cur.execute(SQL("CREATE SCHEMA IF NOT EXISTS {};").format(Identifier(schema_name)))


def list_sql_files() -> list[Path]:
    return sorted(Path(SQL_DIR).glob("*.sql"))


def read_sql_file(sql_file: Path) -> str:
    print(f"Reading SQL file: {sql_file.name}")
    return sql_file.read_text(encoding="utf-8")


def execute_sql_in_schema(conn: Any, schema_name: str, sql_text: str) -> None:
    with conn.cursor() as cur:
        cur.execute(SQL("SET search_path TO {}, public;").format(Identifier(schema_name)))
        cur.execute(sql_text)


def sync_sql_file_to_schema(conn: Any, schema_name: str, sql_file: Path) -> None:
    sql_text = read_sql_file(sql_file)
    execute_sql_in_schema(conn, schema_name, sql_text)
    print(f"Ensured objects from {sql_file.name} exist in schema '{schema_name}'")


def setup_postgres_tables() -> None:
    sql_files = list_sql_files()
    if not sql_files:
        print(f"No SQL files found in {SQL_DIR}")
        return

    with connect(pg_dsn()) as conn:
        ensure_schema(conn, POSTGRES_SCHEMA)
        for sql_file in sql_files:
            print(f"Processing SQL file: {sql_file.name}")
            sync_sql_file_to_schema(conn, POSTGRES_SCHEMA, sql_file)
            print(f"Finished processing {sql_file.name}\n")
        conn.commit()


if __name__ == "__main__":
    wait_for_postgres()
    setup_postgres_tables()
    qdrant = get_qdrant_client()

    with connect(pg_dsn()) as conn:
        check_postgres()
    
    ensure_collection(qdrant)