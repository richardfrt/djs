"""Motor matemático local de recomendación."""

from urllib.parse import quote_plus

import numpy as np

from database import obtener_dataset


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
    """Devuelve las tres mejores canciones, sin repetir el artista de entrada."""
    dataset = obtener_dataset()
    artista_normalizado = str(artista_input).strip().casefold()
    candidatos = dataset[dataset["genero"] == genero].copy()

    if estructura_deseada != "Indiferente":
        candidatos = candidatos[candidatos["estructura"] == estructura_deseada]
    if artista_normalizado:
        candidatos = candidatos[candidatos["artista"].str.casefold() != artista_normalizado]

    bpm_origen, key_origen = _patron_referencia(genero, energia_deseada)
    if candidatos.empty:
        return candidatos, {"bpm": bpm_origen, "key_camelot": key_origen}

    candidatos["distancia_energia"] = (candidatos["energia"] - energia_deseada).abs()
    candidatos["distancia_bpm"] = (candidatos["bpm"] - bpm_origen).abs()
    candidatos["distancia_camelot"] = candidatos["key_camelot"].map(
        lambda clave: _distancia_camelot(key_origen, clave)
    )
    candidatos["puntuacion"] = (
        100
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
