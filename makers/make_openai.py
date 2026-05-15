import os
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
