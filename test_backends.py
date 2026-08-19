"""Testa Hermes e OpenClaw em sequência."""
import truststore
truststore.inject_into_ssl()

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

PERGUNTAS = [
    "Diga olá em uma frase curta.",
    "Que dia é hoje?",
    "O que é Azure Functions em uma frase?",
]

VOICE_PROMPT = "Responda em no máximo 2 frases curtas, sem markdown."


def cabecalho(nome: str, url: str) -> None:
    print(f"\n{'=' * 70}\n  {nome}  @  {url}\n{'=' * 70}")


def linha(i: int, dt: float, q: str, a: str) -> None:
    print(f"[{i}] ⏱️ {dt:.2f}s")
    print(f"    Q: {q}")
    print(f"    A: {a[:300]}\n")


def health_hermes(url: str, headers: dict) -> bool:
    """Hermes expõe /health (root), não /v1/health."""
    root = url.rstrip("/v1")
    try:
        return httpx.get(f"{root}/health", headers=headers, timeout=5).status_code == 200
    except Exception:
        return False


def testar_hermes() -> None:
    url   = os.getenv("HERMES_BASE_URL")
    token = os.getenv("HERMES_TOKEN", "")
    model = os.getenv("HERMES_MODEL", "hermes")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    cabecalho("HERMES", url)
    if not health_hermes(url, headers):
        print("❌ Health falhou\n")
        return
    print("✅ Health OK\n")

    for i, q in enumerate(PERGUNTAS, 1):
        t0 = time.perf_counter()
        try:
            texto = _falar_hermes(url, headers, model, q)
            linha(i, time.perf_counter() - t0, q, texto)
        except Exception as e:
            print(f"[{i}] ❌ {e}\n")


def testar_openclaw() -> None:
    url   = os.getenv("OPENCLAW_BASE_URL")
    token = os.getenv("OPENCLAW_TOKEN", "")
    model = os.getenv("OPENCLAW_MODEL", "openclaw/default")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    cabecalho("OPENCLAW", url)
    try:
        ok = httpx.get(f"{url}/models", headers=headers, timeout=5).status_code == 200
    except Exception as e:
        print(f"❌ Liveness falhou: {e}\n")
        return
    if not ok:
        print("❌ /v1/models não retornou 200\n")
        return
    print("✅ Liveness OK\n")

    for i, q in enumerate(PERGUNTAS, 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                f"{url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": VOICE_PROMPT},
                        {"role": "user",   "content": q},
                    ],
                    "user": "test-session",
                    "stream": False,
                },
                timeout=60,
            )
            r.raise_for_status()
            texto = (r.json()["choices"][0]["message"].get("content") or "").strip()
            linha(i, time.perf_counter() - t0, q, texto)
        except Exception as e:
            print(f"[{i}] ❌ {e}\n")


def _falar_hermes(url: str, headers: dict, model: str, pergunta: str) -> str:
    """Tenta /v1/responses; se 404, cai pra /v1/chat/completions."""
    # Tentativa 1: /v1/responses (persistente, mais moderno)
    try:
        r = httpx.post(
            f"{url}/responses",
            headers=headers,
            json={
                "model": model,
                "input": pergunta,
                "conversation": "test-session",
                "instructions": VOICE_PROMPT,
                "store": True,
            },
            timeout=60,
        )
        if r.status_code != 404:
            r.raise_for_status()
            data = r.json()
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for parte in item.get("content", []):
                        if parte.get("type") == "output_text":
                            return parte.get("text", "")
            return data.get("output_text", "") or ""
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise

    # Fallback: /v1/chat/completions (stateless, universal)
    r = httpx.post(
        f"{url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": VOICE_PROMPT},
                {"role": "user",   "content": pergunta},
            ],
            "user": "test-session",
            "stream": False,
        },
        timeout=60,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


if __name__ == "__main__":
    print("🧪 Testando os dois backends...\n")
    testar_hermes()
    testar_openclaw()
    print(f"\n{'=' * 70}\n  ✅ Concluído\n{'=' * 70}")
