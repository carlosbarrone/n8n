import json
import os
import time

from uuid import uuid4
from typing import Any, Callable
from mistralai.client import Mistral
from qdrant_client import QdrantClient

EXPERIENCES_DATA_PATH = "data/experiences.json"
SKILLS_DATA_PATH = "data/skills.json"
EDUCATION_DATA_PATH = "data/education.json"
CERTIFICATIONS_DATA_PATH = "data/certifications.json"
LANGUAGES_DATA_PATH = "data/languages.json"
SUMMARY_INFO_DATA_PATH = "data/summary_info.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
MISTRAL_EMBEDDING_MODEL = "mistral-embed"
MAX_CHUNK_CHARS = int(os.getenv("QDRANT_MAX_CHUNK_CHARS", "700"))
CHUNK_OVERLAP_CHARS = int(os.getenv("QDRANT_CHUNK_OVERLAP_CHARS", "120"))

_mistral_client: Mistral | None = None


def get_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is not set.")
        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


def open_json_file(path: str) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def get_embedding(text: str, max_retries: int = 6) -> list[float]:
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = get_client().embeddings.create(model=MISTRAL_EMBEDDING_MODEL, inputs=[text])
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1 and "429" in str(e):
                print(f"Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def split_text_into_chunks(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return [normalized]

    words = normalized.split()
    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])
        if current_words and len(candidate) > max_chars:
            chunk = " ".join(current_words)
            chunks.append(chunk)

            overlap_words: list[str] = []
            overlap_length = 0
            for existing_word in reversed(current_words):
                next_length = overlap_length + len(existing_word) + (1 if overlap_words else 0)
                if next_length > overlap_chars:
                    break
                overlap_words.insert(0, existing_word)
                overlap_length = next_length

            current_words = overlap_words + [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def experience_to_text(experience: dict[str, Any]) -> str:
    skills = ", ".join(experience.get("skills_used", []))
    return (
        f"Role: {experience.get('role', '')}. "
        f"Company: {experience.get('company', '')}. "
        f"Location: {experience.get('location', '')}. "
        f"Period: {experience.get('start_date', '')} to {experience.get('end_date', '')}. "
        f"Description: {experience.get('description', '')}. "
        f"Skills: {skills}."
    )


def skill_to_text(skill: dict[str, Any]) -> str:
    return (
        f"Skill: {skill.get('skill', '')}. "
        f"Category: {skill.get('category', '')}. "
        f"Proficiency: {skill.get('proficiency', '')}. "
        f"Years of experience: {skill.get('experience_years', '')}. "
        f"Brand: {skill.get('brand', '')}."
    )


def education_to_text(education: dict[str, Any]) -> str:
    return (
        f"Institution: {education.get('institution', '')}. "
        f"Degree: {education.get('degree', '')}. "
        f"Field of study: {education.get('field_of_study', '')}. "
        f"Period: {education.get('start_date', '')} to {education.get('end_date', '')}. "
        f"Grade: {education.get('grade', '')}. "
        f"Description: {education.get('description', '')}."
    )


def certification_to_text(certification: dict[str, Any]) -> str:
    return (
        f"Certification: {certification.get('name', '')}. "
        f"Issuer: {certification.get('issuer', '')}. "
        f"Date: {certification.get('date', '')}. "
        f"Credential ID: {certification.get('credential_id', '')}. "
        f"Credential URL: {certification.get('credential_url', '')}."
    )


def language_to_text(language: dict[str, Any]) -> str:
    return (
        f"Language: {language.get('language', '')}. "
        f"Proficiency: {language.get('proficiency', '')}."
    )


def summary_info_to_text(summary_info: dict[str, Any]) -> str:
    current_focus = ", ".join(summary_info.get("current_focus", []))
    core_domains = ", ".join(summary_info.get("core_domains", []))
    top_technologies = ", ".join(summary_info.get("top_technologies", []))
    education_summary = "; ".join(summary_info.get("education_summary", []))
    certification_summary = "; ".join(summary_info.get("certification_summary", []))
    languages = ", ".join(summary_info.get("languages", []))
    return (
        f"Full name: {summary_info.get('full_name', '')}. "
        f"Headline: {summary_info.get('headline', '')}. "
        f"Location: {summary_info.get('location', '')}. "
        f"Years of experience: {summary_info.get('years_experience', '')}. "
        f"Professional summary: {summary_info.get('professional_summary', '')}. "
        f"Current role: {summary_info.get('current_role', '')}. "
        f"Current focus: {current_focus}. "
        f"Core domains: {core_domains}. "
        f"Top technologies: {top_technologies}. "
        f"Education: {education_summary}. "
        f"Certifications: {certification_summary}. "
        f"Languages: {languages}."
    )


def build_experience_payload(
    experience: dict[str, Any],
    chunk_text: str,
    document_id: str,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "content": chunk_text,
        "company": experience.get("company", ""),
        "role": experience.get("role", ""),
        "location": experience.get("location", ""),
        "start_date": experience.get("start_date", ""),
        "end_date": experience.get("end_date", ""),
        "skills_used": experience.get("skills_used", []),
        "source_type": "experience",
    }


def build_summary_info_payload(
    summary_info: dict[str, Any],
    chunk_text: str,
    document_id: str,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "content": chunk_text,
        "full_name": summary_info.get("full_name", ""),
        "headline": summary_info.get("headline", ""),
        "location": summary_info.get("location", ""),
        "years_experience": summary_info.get("years_experience", ""),
        "current_role": summary_info.get("current_role", ""),
        "current_focus": summary_info.get("current_focus", []),
        "core_domains": summary_info.get("core_domains", []),
        "top_technologies": summary_info.get("top_technologies", []),
        "languages": summary_info.get("languages", []),
        "source_type": "summary_info",
    }


def push_records_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    to_text: Callable[[dict[str, Any]], str],
) -> None:
    for record in records:
        text = to_text(record)
        vector = get_embedding(text)
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid4()),
                    "vector": vector,
                    "payload": record,
                }
            ],
        )
    print(f"Pushed {len(records)} record(s) to {collection_name}")


def push_chunked_records_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    to_text: Callable[[dict[str, Any]], str],
    build_payload: Callable[[dict[str, Any], str, str, int, int], dict[str, Any]],
) -> None:
    total_chunks = 0
    for record in records:
        document_id = str(uuid4())
        text = to_text(record)
        chunks = split_text_into_chunks(text)
        for chunk_index, chunk_text in enumerate(chunks):
            vector = get_embedding(chunk_text)
            payload = build_payload(record, chunk_text, document_id, chunk_index, len(chunks))
            client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": str(uuid4()),
                        "vector": vector,
                        "payload": payload,
                    }
                ],
            )
            total_chunks += 1
    print(f"Pushed {len(records)} record(s) as {total_chunks} chunk(s) to {collection_name}")


if __name__ == "__main__":
    experiences = open_json_file(EXPERIENCES_DATA_PATH)
    skills = open_json_file(SKILLS_DATA_PATH)
    education = open_json_file(EDUCATION_DATA_PATH)
    certifications = open_json_file(CERTIFICATIONS_DATA_PATH)
    languages = open_json_file(LANGUAGES_DATA_PATH)
    summary_info = open_json_file(SUMMARY_INFO_DATA_PATH)

    qdrant_client = QdrantClient(url=QDRANT_URL)
    push_chunked_records_to_qdrant(qdrant_client, "experiences", experiences, experience_to_text, build_experience_payload)
    push_records_to_qdrant(qdrant_client, "skills", skills, skill_to_text)
    push_records_to_qdrant(qdrant_client, "education", education, education_to_text)
    push_records_to_qdrant(qdrant_client, "certifications", certifications, certification_to_text)
    push_records_to_qdrant(qdrant_client, "languages", languages, language_to_text)
    push_chunked_records_to_qdrant(qdrant_client, "summary_info", summary_info, summary_info_to_text, build_summary_info_payload)
    