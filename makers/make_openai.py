import io
import os
import json

from openai import AsyncOpenAI

DEFAULT_CONCURRENCY = 2


def make_client(spec: dict, timeout: int) -> AsyncOpenAI:
    api_key = os.getenv(spec["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"variável de ambiente {spec['api_key_env']} não definida — verifique o .env"
        )
    return AsyncOpenAI(api_key=api_key, timeout=timeout)


async def call_once(
    client, name: str, spec: dict, question: str,
    system_prompt: str, temperature: float, max_tokens: int,
) -> tuple[str, str]:
    resp = await client.chat.completions.create(
        model=spec["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    provider = f"openai:{getattr(resp, 'model', spec['model'])}"
    return text, provider


async def close_client(client) -> None:
    await client.close()


# --------------------------------------------------------------------------
# Batch API — 50% de desconto, completion_window=24h (única opção).
# Fluxo: upload JSONL via files.create → batches.create → poll → files.content.
# --------------------------------------------------------------------------

def supports_batch() -> bool:
    return True


async def submit_batch(
    client, name: str, spec: dict, items: list[dict],
    system_prompt: str, temperature: float, max_tokens: int,
) -> str:
    """Faz upload do JSONL e cria o batch. Retorna batch_id."""
    lines = []
    for it in items:
        lines.append(json.dumps({
            "custom_id": f"item_{it['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": spec["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": it["question"]},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        }, ensure_ascii=False))
    jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

    file_obj = await client.files.create(
        file=("batch.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    batch = await client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


async def poll_batch(client, batch_id: str) -> dict:
    batch = await client.batches.retrieve(batch_id)
    if batch.status == "completed":
        return {"status": "done", "raw": batch}
    if batch.status in ("failed", "expired", "cancelled", "cancelling"):
        return {"status": "failed", "raw": batch}
    return {"status": "pending", "raw": batch}


async def fetch_batch(client, batch_id: str) -> list[dict]:
    """Baixa output_file_id e parseia JSONL."""
    batch = await client.batches.retrieve(batch_id)
    if batch.status != "completed":
        return []
    if not batch.output_file_id:
        return []

    response = await client.files.content(batch.output_file_id)
    if hasattr(response, "aread"):
        content_bytes = await response.aread()
    elif hasattr(response, "content"):
        content_bytes = response.content
    else:
        content_bytes = response.read()

    out = []
    for line in content_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        custom_id = r["custom_id"]
        if r.get("error"):
            out.append({
                "custom_id": custom_id,
                "generation": None,
                "provider_used": "openai",
                "error": str(r["error"]),
            })
            continue
        body = r["response"]["body"]
        text = (body["choices"][0]["message"].get("content") or "").strip()
        out.append({
            "custom_id": custom_id,
            "generation": text,
            "provider_used": f"openai:{body.get('model', 'unknown')}",
            "error": None,
        })
    return out
