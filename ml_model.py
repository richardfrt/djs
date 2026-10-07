"""Motor de recomendación basado en BPM, energía y Rueda Camelot."""

import pandas as pd

from database import cargar_datos


def _parsear_clave(clave):
    """Convierte una clave Camelot en número y letra."""
    texto = str(clave).strip().upper()
    if len(texto) < 2:
        return None

    letra = texto[-1]
    numero_texto = texto[:-1]
    if letra not in {"A", "B"} or not numero_texto.isdigit():
        return None

    numero = int(numero_texto)
    if numero < 1 or numero > 12:
        return None
    return numero, letra


def es_clave_compatible(clave_origen, clave_destino):
    """Aplica las reglas estrictas de compatibilidad de la Rueda Camelot."""
    origen = _parsear_clave(clave_origen)
    destino = _parsear_clave(clave_destino)
    if origen is None or destino is None:
        return False

    numero_origen, letra_origen = origen
    numero_destino, letra_destino = destino

    if origen == destino:
        return True

    if numero_origen == numero_destino and letra_origen != letra_destino:
        return True

    misma_letra = letra_origen == letra_destino
    diferencia_circular = abs(numero_origen - numero_destino)
    diferencia_circular = min(diferencia_circular, 12 - diferencia_circular)
    return misma_letra and diferencia_circular == 1


def recomendar_matches(
    artista_usuario,
    genero,
    energia,
    estructura,
    bpm_orig,
    key_orig,
):
    """Devuelve hasta tres canciones compatibles con el patrón introducido."""
    dataframe = cargar_datos()

    artista_usuario = str(artista_usuario).strip()
    if artista_usuario:
        dataframe = dataframe[
            ~dataframe["artista"].astype(str).str.contains(
                artista_usuario,
                case=False,
                regex=False,
                na=False,
            )
        ]

    if genero != "Indiferente":
        dataframe = dataframe[dataframe["genero"] == genero]

    if estructura != "Indiferente":
        dataframe = dataframe[dataframe["estructura"] == estructura]

    minimo_bpm = int(bpm_orig) - 3
    maximo_bpm = int(bpm_orig) + 3
    dataframe = dataframe[
        dataframe["bpm"].between(minimo_bpm, maximo_bpm)
    ]

    dataframe = dataframe[
        dataframe["key_camelot"].map(
            lambda clave: es_clave_compatible(key_orig, clave)
        )
    ]

    dataframe = dataframe.copy()
    dataframe["distancia_energia"] = (
        dataframe["energia"] - int(energia)
    ).abs()

    return dataframe.sort_values(
        by="distancia_energia",
        ascending=True,
    ).head(3)
