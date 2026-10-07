"""Motor de recomendación musical basado en OpenAI."""

import json

import pandas as pd
from openai import OpenAI, OpenAIError

from database import cargar_datos, guardar_datos


COLUMNAS_CANCIONES = [
    "titulo",
    "artista",
    "genero",
    "bpm",
    "key_camelot",
    "energia",
    "estructura",
    "sello",
]


def analizar_referencia(api_key, cancion, artista):
    """Obtiene el BPM y la clave Camelot de una canción mediante OpenAI."""
    if not api_key:
        raise ValueError(
            "Introduce la clave de OpenAI para poder analizar la canción."
        )

    cliente = OpenAI(api_key=api_key)
    try:
        respuesta = cliente.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un analizador musical. Tu único trabajo es "
                        'devolver el BPM exacto (int) y la Clave Camelot '
                        '(string, ej. "8A") de la canción dada. Devuelve '
                        'SOLO un JSON con las claves "bpm" y "key_camelot". '
                        "Si no estás 100% seguro, haz tu mejor estimación "
                        "basada en el género."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Canción: {cancion} - Artista: {artista}",
                },
            ],
        )
    except OpenAIError as error:
        raise ValueError(
            f"No se pudo analizar la referencia con OpenAI: {error}"
        ) from error

    contenido = respuesta.choices[0].message.content
    if not contenido:
        raise ValueError("OpenAI devolvió una respuesta vacía.")

    try:
        datos = json.loads(contenido)
        bpm = int(datos["bpm"])
        key_camelot = str(datos["key_camelot"]).strip().upper()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "OpenAI no devolvió un análisis válido de BPM y clave Camelot."
        ) from error

    return bpm, key_camelot


def recomendar_matches(
    cancion,
    artista_usuario,
    genero,
    energia,
    estructura,
    api_key,
):
    """Obtiene cinco recomendaciones de OpenAI y las persiste localmente."""
    bpm_orig, key_orig = analizar_referencia(
        api_key,
        cancion,
        artista_usuario,
    )

    prompt = (
        f"Eres un A&R de élite. Busca 5 joyas underground para mezclar con "
        f"{cancion} de {artista_usuario}. REGLAS: 1. Cero comercial, usa "
        "sellos de culto (Afterlife, Drumcode, Solid Grooves...). 2. "
        f"Artista distinto a {artista_usuario}. 3. Género: {genero}, "
        f"Energía: {energia}, Estructura: {estructura}. 4. BPM entre "
        f"{bpm_orig - 3} y {bpm_orig + 3}. 5. Clave armónicamente compatible "
        f"con {key_orig}. Devuelve exactamente 5 canciones reales y solo un "
        'JSON con esta estructura: {"canciones": [{"titulo": "", '
        '"artista": "", "genero": "", "bpm": 125, "key_camelot": "8A", '
        '"energia": 8, "estructura": "", "sello": ""}]}'
    )

    cliente = OpenAI(api_key=api_key)
    try:
        respuesta = cliente.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un A&R de élite. Devuelve únicamente un objeto "
                        "JSON válido con la clave 'canciones'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except OpenAIError as error:
        raise ValueError(
            f"No se pudieron obtener recomendaciones de OpenAI: {error}"
        ) from error

    contenido = respuesta.choices[0].message.content
    if not contenido:
        raise ValueError("OpenAI devolvió una respuesta vacía.")

    try:
        datos = json.loads(contenido)
        canciones = datos["canciones"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "OpenAI no devolvió el JSON esperado con la clave 'canciones'."
        ) from error

    if not isinstance(canciones, list) or len(canciones) != 5:
        raise ValueError(
            "OpenAI debe devolver exactamente 5 canciones válidas."
        )

    for cancion_recomendada in canciones:
        if (
            not isinstance(cancion_recomendada, dict)
            or set(cancion_recomendada) != set(COLUMNAS_CANCIONES)
        ):
            raise ValueError(
                "Una recomendación no contiene exactamente las columnas "
                "requeridas."
            )

    df_nuevo = pd.DataFrame(canciones, columns=COLUMNAS_CANCIONES)
    df_nuevo["bpm"] = pd.to_numeric(df_nuevo["bpm"], errors="raise").astype(int)
    df_nuevo["energia"] = (
        pd.to_numeric(df_nuevo["energia"], errors="raise").astype(int)
    )

    try:
        df_actual = cargar_datos()
        df_completo = pd.concat(
            [df_actual, df_nuevo],
            ignore_index=True,
        ).drop_duplicates(
            subset=["titulo", "artista"],
            keep="last",
        )
        guardar_datos(df_completo)
    except Exception:
        guardar_datos(df_nuevo)

    return df_nuevo, bpm_orig, key_orig
