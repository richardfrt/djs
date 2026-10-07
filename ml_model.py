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
    "motivo_del_match",
]


def _parsear_clave(clave):
    texto = str(clave).strip().upper()
    if len(texto) < 2 or texto[-1] not in {"A", "B"}:
        return None
    numero = texto[:-1]
    if not numero.isdigit() or not 1 <= int(numero) <= 12:
        return None
    return int(numero), texto[-1]


def _claves_compatibles(clave_origen, clave_destino):
    origen = _parsear_clave(clave_origen)
    destino = _parsear_clave(clave_destino)
    if origen is None or destino is None:
        return False

    numero_origen, letra_origen = origen
    numero_destino, letra_destino = destino
    diferencia = abs(numero_origen - numero_destino)
    diferencia = min(diferencia, 12 - diferencia)
    return (
        origen == destino
        or (
            numero_origen == numero_destino
            and letra_origen != letra_destino
        )
        or (letra_origen == letra_destino and diferencia == 1)
    )


def analizar_referencia(api_key, cancion, artista):
    """Obtiene BPM, clave y contexto sonoro mediante OpenAI."""
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
                        "Eres un analizador musical. Extrae el BPM exacto "
                        "(int), la Clave Camelot (string, ej. \"8A\"), "
                        "groove_style, elementos_sonoros y momento_pista. "
                        'Devuelve SOLO un JSON con las claves "bpm", '
                        '"key_camelot", "groove_style", "elementos_sonoros" '
                        'y "momento_pista". '
                        "Si no estás 100% seguro, haz tu mejor estimación "
                        "basada en el género y el contexto musical."
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
        groove_style = str(datos["groove_style"]).strip()
        elementos_sonoros = str(datos["elementos_sonoros"]).strip()
        momento_pista = str(datos["momento_pista"]).strip()
        if not groove_style or not elementos_sonoros or not momento_pista:
            raise ValueError("Faltan campos de contexto sonoro.")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "OpenAI no devolvió un análisis musical completo y válido."
        ) from error

    return bpm, key_camelot, groove_style, elementos_sonoros, momento_pista


def recomendar_matches(
    cancion,
    artista_usuario,
    genero,
    energia,
    estructura,
    api_key,
):
    """Obtiene cinco recomendaciones de OpenAI y las persiste localmente."""
    (
        bpm_orig,
        key_orig,
        groove_style,
        elementos_sonoros,
        momento_pista,
    ) = analizar_referencia(
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
        f"con {key_orig}. La canción de referencia tiene un groove "
        f"[{groove_style}], destaca por [{elementos_sonoros}] y pertenece "
        f"al contexto de [{momento_pista}]. Las 5 joyas underground que "
        "recomiendes DEBEN pertenecer exactamente a este mismo ecosistema "
        "sonoro y cultural. No mezcles vibras (ej. no pongas techno oscuro "
        "si la referencia es house alegre playero). Devuelve exactamente 5 "
        "canciones reales y solo un JSON con esta estructura: "
        '{"canciones": [{"titulo": "", '
        '"artista": "", "genero": "", "bpm": 125, "key_camelot": "8A", '
        '"energia": 8, "estructura": "", "sello": "", '
        '"motivo_del_match": "Groove y vibra compatibles por compartir '
        'contexto sonoro underground."}]}'
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
    minimo_bpm = bpm_orig - 3
    maximo_bpm = bpm_orig + 3
    df_nuevo["artista"] = df_nuevo["artista"].astype(str).str.strip()
    df_nuevo["genero"] = df_nuevo["genero"].astype(str).str.strip()
    df_nuevo["estructura"] = df_nuevo["estructura"].astype(str).str.strip()
    df_nuevo["key_camelot"] = (
        df_nuevo["key_camelot"].astype(str).str.strip().str.upper()
    )
    df_nuevo = df_nuevo[
        (df_nuevo["artista"].str.casefold() != str(artista_usuario).strip().casefold())
        & df_nuevo["bpm"].between(minimo_bpm, maximo_bpm)
        & (df_nuevo["genero"] == genero)
        & df_nuevo["energia"].between(1, 10)
        & (
            (estructura == "Indiferente")
            | (df_nuevo["estructura"] == estructura)
        )
        & df_nuevo["key_camelot"].map(
            lambda clave: _claves_compatibles(key_orig, clave)
        )
    ].copy()
    if len(df_nuevo) != 5:
        raise ValueError(
            "OpenAI no devolvió cinco canciones que cumplan los filtros "
            "matemáticos solicitados."
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
