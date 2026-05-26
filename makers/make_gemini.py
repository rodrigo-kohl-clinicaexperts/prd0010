import asyncio
import io
import json
import os
import tempfile

from google import genai
from google.genai import types

DEFAULT_CONCURRENCY = 2


def make_client(spec: dict, timeout: int) -> genai.Client:
    api_key = os.getenv(spec["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"variável de ambiente {spec['api_key_env']} não definida — verifique o .env"
        )
    return genai.Client(api_key=api_key)


async def call_once(
    client, name: str, spec: dict, question: str,
    system_prompt: str, temperature: float, max_tokens: int,
) -> tuple[str, str]:
    resp = await client.aio.models.generate_content(
        model=spec["model"],
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = (resp.text or "").strip()
    provider = f"gemini:{spec['model']}"
    return text, provider


async def close_client(client) -> None:
    pass  # google-genai não tem close explícito


# --------------------------------------------------------------------------
# Batch API — 50% de desconto, SLA 48h. Fluxo: upload JSONL via files.upload
# → batches.create(src=file.name) → poll → files.download(file=result_file).
# SDK é síncrono em batches/files; envolvemos com asyncio.to_thread.
# --------------------------------------------------------------------------

def supports_batch() -> bool:
    return True


async def submit_batch(
    client, name: str, spec: dict, items: list[dict],
    system_prompt: str, temperature: float, max_tokens: int,
) -> str:
    """Faz upload do JSONL e cria o batch. Retorna o batch name (ID)."""
    lines = []
    for it in items:
        lines.append(json.dumps({
            "key": f"item_{it['id']}",
            "request": {
                "contents": [{"parts": [{"text": it["question"]}]}],
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "generation_config": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            },
        }, ensure_ascii=False))
    jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

    # google-genai files.upload exige path em disco → tempfile
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".jsonl", delete=False
    ) as f:
        f.write(jsonl_bytes)
        tmp_path = f.name

    try:
        uploaded = await asyncio.to_thread(
            client.files.upload,
            file=tmp_path,
            config=types.UploadFileConfig(
                display_name=f"batch_{name}",
                mime_type="application/jsonl",
            ),
        )
    finally:
        os.unlink(tmp_path)

    job = await asyncio.to_thread(
        client.batches.create,
        model=spec["model"],
        src=uploaded.name,
        config=types.CreateBatchJobConfig(display_name=f"batch_{name}"),
    )
    return job.name


async def poll_batch(client, batch_id: str) -> dict:
    job = await asyncio.to_thread(client.batches.get, name=batch_id)
    state = str(job.state)
    if "SUCCEEDED" in state:
        return {"status": "done", "raw": job}
    if "FAILED" in state or "CANCELLED" in state or "EXPIRED" in state:
        return {"status": "failed", "raw": job}
    return {"status": "pending", "raw": job}


async def fetch_batch(client, batch_id: str) -> list[dict]:
    job = await asyncio.to_thread(client.batches.get, name=batch_id)
    if "SUCCEEDED" not in str(job.state):
        return []

    # job.dest é BatchJobDestination; pode ser file_name ou inlined_responses
    dest = job.dest
    if dest is None or not getattr(dest, "file_name", None):
        return []
    result_file_name = dest.file_name
    content_bytes = await asyncio.to_thread(client.files.download, file=result_file_name)

    out = []
    for line in content_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        custom_id = r.get("key", "")
        if "error" in r and r["error"]:
            out.append({
                "custom_id": custom_id,
                "generation": None,
                "provider_used": "gemini",
                "error": str(r["error"]),
            })
            continue
        try:
            text = r["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as e:
            out.append({
                "custom_id": custom_id,
                "generation": None,
                "provider_used": "gemini",
                "error": f"parse_error: {e}",
            })
            continue
        out.append({
            "custom_id": custom_id,
            "generation": text,
            "provider_used": "gemini",
            "error": None,
        })
    return out
