"""Motor matemático local de recomendación."""

import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import quote_plus

import numpy as np

from database import obtener_dataset


def _normalizar_texto(texto):
    """Normaliza nombres para comparar artistas sin depender de mayúsculas o acentos."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(caracter for caracter in texto if not unicodedata.combining(caracter))
    return re.sub(r"[^a-z0-9]+", " ", texto.casefold()).strip()


def _artistas_coinciden(artista_entrada, artista_dataset):
    """Detecta coincidencias exactas y variantes muy similares del artista."""
    entrada = _normalizar_texto(artista_entrada)
    candidato = _normalizar_texto(artista_dataset)
    if not entrada or not candidato:
        return False
    if entrada == candidato:
        return True

    # Evita recomendar variantes como "Artist" y "Artist Remix/Live".
    if len(entrada) >= 4 and (entrada in candidato or candidato in entrada):
        return True
    return SequenceMatcher(None, entrada, candidato).ratio() >= 0.82


def _normalizar_camelot(clave):
    try:
        numero = int(str(clave)[:-1])
        tipo = str(clave)[-1].upper()
        if tipo not in {"A", "B"} or not 1 <= numero <= 12:
            raise ValueError
        return numero, tipo
    except (TypeError, ValueError):
        return 1, "A"


def _distancia_camelot(origen, destino):
    numero_origen, tipo_origen = _normalizar_camelot(origen)
    numero_destino, tipo_destino = _normalizar_camelot(destino)
    distancia = abs(numero_origen - numero_destino)
    return min(distancia, 12 - distancia) + int(tipo_origen != tipo_destino)


def _patron_referencia(genero, energia):
    bases_bpm = {"Techno": 128, "Tech House": 126, "Melodic": 124}
    bases_key = {"Techno": 8, "Tech House": 5, "Melodic": 9}
    bpm = bases_bpm[genero] + int(energia) - 6
    numero = int(np.clip(bases_key[genero] + round((int(energia) - 6) / 3), 1, 12))
    tipo = "B" if genero == "Melodic" else "A"
    return bpm, f"{numero}{tipo}"


def recomendar_matches(cancion_input, artista_input, genero, energia_deseada, estructura_deseada):
    """Recomienda por patrones, sin buscar la canción de entrada en el dataset.

    ``cancion_input`` se conserva como contexto de la interfaz, pero no se usa
    para filtrar ni exigir que exista en la base local.
    """
    del cancion_input
    dataset = obtener_dataset()
    candidatos = dataset[dataset["genero"] == genero].copy()

    if estructura_deseada != "Indiferente":
        candidatos = candidatos[candidatos["estructura"] == estructura_deseada]
    if str(artista_input).strip():
        candidatos = candidatos[
            ~candidatos["artista"].map(
                lambda artista: _artistas_coinciden(artista_input, artista)
            )
        ]

    bpm_origen, key_origen = _patron_referencia(genero, energia_deseada)
    if candidatos.empty:
        return candidatos, {"bpm": bpm_origen, "key_camelot": key_origen}

    candidatos["distancia_energia"] = (candidatos["energia"] - energia_deseada).abs()
    candidatos["distancia_bpm"] = (candidatos["bpm"] - bpm_origen).abs()
    candidatos["distancia_camelot"] = candidatos["key_camelot"].map(
        lambda clave: _distancia_camelot(key_origen, clave)
    )
    candidatos["puntuacion"] = (
        100  # Todos los candidatos ya tienen afinidad de género tras el filtro.
        - candidatos["distancia_energia"] * 12
        - candidatos["distancia_bpm"] * 1.8
        - candidatos["distancia_camelot"] * 5
    )
    candidatos["youtube_url"] = candidatos.apply(
        lambda fila: "https://www.youtube.com/results?search_query="
        + quote_plus(f"{fila['artista']} {fila['titulo']}"),
        axis=1,
    )
    candidatos["compatibilidad"] = candidatos.apply(
        lambda fila: (
            f"Energía ±{int(fila['distancia_energia'])} · "
            f"BPM ±{int(fila['distancia_bpm'])} · "
            f"Camelot {fila['key_camelot']} frente a {key_origen}"
        ),
        axis=1,
    )
    return (
        candidatos.sort_values(
            ["puntuacion", "distancia_energia", "distancia_bpm"],
            ascending=[False, True, True],
        ).head(3),
        {"bpm": bpm_origen, "key_camelot": key_origen},
    )
