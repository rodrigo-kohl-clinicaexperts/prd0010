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
