import os
from openai import AsyncOpenAI

DEFAULT_CONCURRENCY = 16


def make_client(spec: dict, timeout: int) -> AsyncOpenAI:
    api_key = os.getenv(spec["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"variável de ambiente {spec['api_key_env']} não definida — verifique o .env"
        )
    return AsyncOpenAI(api_key=api_key, base_url=spec["base_url"], timeout=timeout)


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
    provider = getattr(resp, "model", None) or "openrouter"
    return text, str(provider)


async def close_client(client) -> None:
    await client.close()


# OpenRouter ainda não expõe Batch API equivalente a OpenAI/Anthropic/Gemini.
def supports_batch() -> bool:
    return False


async def submit_batch(*args, **kwargs):
    raise NotImplementedError("OpenRouter não tem batch API")


async def poll_batch(*args, **kwargs):
    raise NotImplementedError("OpenRouter não tem batch API")


async def fetch_batch(*args, **kwargs):
    raise NotImplementedError("OpenRouter não tem batch API")
