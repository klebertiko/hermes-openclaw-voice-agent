"""
Voice Agent — fala com Hermes E OpenClaw.
Prefixo "Hermes, ..." ou "OpenClaw, ..." direciona pra um agente só.
Sem prefixo: ambos respondem em paralelo.
"""
import truststore
truststore.inject_into_ssl()

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import edge_tts
import httpx
import miniaudio
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# ============================================================
# CONFIGURAÇÃO
# ============================================================
load_dotenv()

SESSION_ID     = os.getenv("SESSION_ID", "voice-session")
EDGE_VOICE     = os.getenv("EDGE_VOICE", "pt-BR-FranciscaNeural")
WHISPER_SIZE   = "small"
SAMPLE_RATE    = 16000
RECORD_SECONDS = 5
EDGE_SR        = 24000
HTTP_TIMEOUT   = 120.0

VOICE_PROMPT = (
    "Você está respondendo por voz. Use no máximo 2 frases curtas em "
    "português brasileiro. Sem markdown, listas, asteriscos, código "
    "ou emojis. Vá direto à resposta."
)

ORDEM_AGENTES = ("hermes", "openclaw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("voice")

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Prefixos de direcionamento na voz: "Hermes, ..." ou "OpenClaw, ..."
PREFIXOS_AGENTE = {
    "hermes":   re.compile(r"^\s*hermes\s*[,:.]?\s+", re.IGNORECASE),
    "openclaw": re.compile(r"^\s*open\s*c[lr]aw\s*[,:.]?\s+", re.IGNORECASE),
}


# ============================================================
# BACKENDS
# ============================================================
def chamar_hermes(texto: str) -> str:
    """Hermes via /v1/chat/completions (endpoint universal)."""
    url   = os.getenv("HERMES_BASE_URL", "http://127.0.0.1:8642/v1")
    token = os.getenv("HERMES_TOKEN", "")
    model = os.getenv("HERMES_MODEL", "hermes")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    r = httpx.post(
        f"{url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": VOICE_PROMPT},
                {"role": "user",   "content": texto},
            ],
            "user": SESSION_ID,
            "stream": False,
        },
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def chamar_openclaw(texto: str) -> str:
    """OpenClaw via /v1/chat/completions com sessão derivada do campo user."""
    url   = os.getenv("OPENCLAW_BASE_URL", "http://127.0.0.1:18789/v1")
    token = os.getenv("OPENCLAW_TOKEN", "")
    model = os.getenv("OPENCLAW_MODEL", "openclaw/default")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    r = httpx.post(
        f"{url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": VOICE_PROMPT},
                {"role": "user",   "content": texto},
            ],
            "user": SESSION_ID,
            "stream": False,
        },
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


CHAMADAS = {
    "hermes":   chamar_hermes,
    "openclaw": chamar_openclaw,
}


def consultar_selecionados(texto: str, alvos: tuple[str, ...]) -> dict[str, str | None]:
    """Dispara apenas os backends pedidos, em paralelo."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(alvos)) as pool:
        futuros = {nome: pool.submit(CHAMADAS[nome], texto) for nome in alvos}
        respostas: dict[str, str | None] = {}
        for nome, fut in futuros.items():
            try:
                respostas[nome] = fut.result(timeout=HTTP_TIMEOUT)
            except Exception as e:
                log.error("❌ %s falhou: %s", nome, e)
                respostas[nome] = None
    log.info("🌐 %s em %.2fs", "+".join(alvos), time.perf_counter() - t0)
    return respostas


# ============================================================
# STT — Whisper
# ============================================================
def carregar_whisper() -> WhisperModel:
    log.info("🔥 Carregando Whisper [%s]...", WHISPER_SIZE)
    t0 = time.perf_counter()
    stt = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    log.info("   pronto em %.1fs", time.perf_counter() - t0)
    return stt


def gravar() -> np.ndarray:
    log.info("🎤 Falando agora (%ds)...", RECORD_SECONDS)
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcrever(stt: WhisperModel, audio: np.ndarray) -> str:
    t0 = time.perf_counter()
    segments, _ = stt.transcribe(
        audio,
        language="pt",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    texto = " ".join(s.text for s in segments).strip()
    log.info("📝 (STT %.2fs) %s", time.perf_counter() - t0, texto)
    return texto


# ============================================================
# TTS — Edge-TTS + miniaudio
# ============================================================
async def _sintetizar(texto: str) -> bytes:
    comm = edge_tts.Communicate(texto, EDGE_VOICE)
    mp3 = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    return mp3


def falar(texto: str) -> None:
    if not texto:
        return
    t0 = time.perf_counter()
    mp3 = asyncio.run(_sintetizar(texto))
    pcm = miniaudio.decode(
        mp3,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=EDGE_SR,
    )
    audio = np.frombuffer(pcm.samples, dtype=np.int16)
    log.info("🔊 (TTS %.2fs)", time.perf_counter() - t0)
    sd.play(audio, samplerate=pcm.sample_rate)
    sd.wait()


# ============================================================
# UTIL
# ============================================================
def limpar(texto: str) -> str:
    texto = _THINK.sub("", texto)
    texto = re.sub(r"[*_`#]+", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def deve_sair(texto: str) -> bool:
    return texto.lower().strip().rstrip(".!?") in {"sair", "tchau", "encerrar", "parar"}


def detectar_alvo(texto: str) -> tuple[tuple[str, ...], str]:
    """Detecta se a fala começa com 'Hermes, ...' ou 'OpenClaw, ...'.
    Retorna (agentes_a_chamar, texto_sem_prefixo)."""
    for nome, regex in PREFIXOS_AGENTE.items():
        if regex.match(texto):
            limpo = regex.sub("", texto).strip()
            return (nome,), limpo
    return ORDEM_AGENTES, texto


def falar_resposta(nome: str, texto: str | None, anunciar: bool) -> None:
    """Fala a resposta. Anuncia o nome só quando os dois respondem."""
    rotulo = nome.capitalize()
    if texto is None:
        falar(f"{rotulo} não respondeu.")
        return
    texto_limpo = limpar(texto)
    if not texto_limpo:
        falar(f"{rotulo} respondeu vazio.")
        return
    log.info("🤖 %s: %s", rotulo, texto_limpo)
    falar(f"{rotulo} diz. {texto_limpo}" if anunciar else texto_limpo)


# ============================================================
# LOOP
# ============================================================
def main() -> None:
    stt = carregar_whisper()
    log.info("🌐 Backends: Hermes + OpenClaw (em paralelo)")
    log.info("🎵 Voz: %s", EDGE_VOICE)
    log.info("✅ Pronto. Ctrl+C para sair.\n")

    falar(
        "Olá! Hermes e OpenClaw estão ouvindo. "
        "Para falar com apenas um, comece a frase com o nome dele."
    )

    try:
        while True:
            audio = gravar()
            texto = transcrever(stt, audio)
            if not texto:
                continue
            if deve_sair(texto):
                falar("Tchau! Até a próxima.")
                break

            alvos, pergunta = detectar_alvo(texto)
            anunciar = len(alvos) > 1
            if not anunciar:
                log.info("🎯 Falando só com %s", alvos[0].capitalize())

            respostas = consultar_selecionados(pergunta, alvos)
            for nome in alvos:
                falar_resposta(nome, respostas[nome], anunciar)
    except KeyboardInterrupt:
        log.info("\nEncerrado.")


if __name__ == "__main__":
    main()
