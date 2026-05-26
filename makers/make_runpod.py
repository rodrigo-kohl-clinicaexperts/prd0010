import os
from openai import AsyncOpenAI

DEFAULT_CONCURRENCY = 4


def make_client(spec: dict, timeout: int) -> AsyncOpenAI:
    api_key = os.getenv(spec["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"variável de ambiente {spec['api_key_env']} não definida — verifique o .env"
        )
    endpoint_id = os.getenv(spec["runpod_endpoint_env"])
    if not endpoint_id:
        raise RuntimeError(
            f"variável de ambiente {spec['runpod_endpoint_env']} não definida — "
            f"coloque o ENDPOINT_ID do RunPod no .env"
        )
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


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
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    base_url = str(client.base_url)
    endpoint_id = base_url.split("/v2/")[1].split("/")[0]
    provider = f"runpod:{endpoint_id}"
    return text, provider


async def close_client(client) -> None:
    await client.close()


# RunPod expõe endpoint OpenAI-compatible mas sem batch processing.
def supports_batch() -> bool:
    return False


async def submit_batch(*args, **kwargs):
    raise NotImplementedError("RunPod não tem batch API")


async def poll_batch(*args, **kwargs):
    raise NotImplementedError("RunPod não tem batch API")


async def fetch_batch(*args, **kwargs):
    raise NotImplementedError("RunPod não tem batch API")
