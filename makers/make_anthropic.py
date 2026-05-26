import os
import anthropic

DEFAULT_CONCURRENCY = 1


def make_client(spec: dict, timeout: int) -> anthropic.AsyncAnthropic:
    api_key = os.getenv(spec["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"variável de ambiente {spec['api_key_env']} não definida — verifique o .env"
        )
    return anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)


async def call_once(
    client, name: str, spec: dict, question: str,
    system_prompt: str, temperature: float, max_tokens: int,
) -> tuple[str, str]:
    resp = await client.messages.create(
        model=spec["model"],
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
    )
    text = resp.content[0].text.strip()
    provider = f"anthropic:{resp.model}"
    return text, provider


async def close_client(client) -> None:
    await client.close()


# --------------------------------------------------------------------------
# Batch API — 50% de desconto, SLA 24h. Sem upload separado: requests vão
# inline em messages.batches.create. Resultados via stream tipado.
# --------------------------------------------------------------------------

def supports_batch() -> bool:
    return True


async def submit_batch(
    client, name: str, spec: dict, items: list[dict],
    system_prompt: str, temperature: float, max_tokens: int,
) -> str:
    """items = [{"id": int, "question": str, ...}, ...]. Retorna batch_id."""
    requests = [
        {
            "custom_id": f"item_{it['id']}",
            "params": {
                "model": spec["model"],
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": it["question"]}],
                "temperature": temperature,
            },
        }
        for it in items
    ]
    batch = await client.messages.batches.create(requests=requests)
    return batch.id


async def poll_batch(client, batch_id: str) -> dict:
    batch = await client.messages.batches.retrieve(batch_id)
    if batch.processing_status == "ended":
        return {"status": "done", "raw": batch}
    return {"status": "pending", "raw": batch}


async def fetch_batch(client, batch_id: str) -> list[dict]:
    """
    Retorna lista normalizada de resultados:
        [{"custom_id", "generation"|None, "provider_used", "error"|None}, ...]
    """
    out = []
    stream = await client.messages.batches.results(batch_id)
    async for r in stream:
        custom_id = r.custom_id
        rtype = r.result.type
        if rtype == "succeeded":
            msg = r.result.message
            text = msg.content[0].text.strip() if msg.content else ""
            out.append({
                "custom_id": custom_id,
                "generation": text,
                "provider_used": f"anthropic:{msg.model}",
                "error": None,
            })
        elif rtype == "errored":
            err = r.result.error
            err_msg = err.error.message if err and err.error else "unknown"
            out.append({
                "custom_id": custom_id,
                "generation": None,
                "provider_used": "anthropic",
                "error": f"errored: {err_msg}",
            })
        else:  # canceled, expired
            out.append({
                "custom_id": custom_id,
                "generation": None,
                "provider_used": "anthropic",
                "error": rtype,
            })
    return out
