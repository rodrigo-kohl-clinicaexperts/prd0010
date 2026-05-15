import os
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
