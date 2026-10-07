"""Motor de recomendación basado en BPM, energía y Rueda Camelot."""

import json

import openai
import pandas as pd

from database import COLUMNAS_REQUERIDAS, cargar_datos, guardar_datos


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


def _artista_excluido(artista, artista_excluir):
    return str(artista_excluir).strip().casefold() in str(artista).casefold()


def buscar_en_nube(
    api_key,
    artista_excluir,
    genero,
    energia,
    estructura,
    bpm_orig,
    key_orig,
    cantidad_necesaria,
):
    """Busca las canciones que faltan y las persiste tras validar el JSON."""
    if not api_key or cantidad_necesaria <= 0:
        return pd.DataFrame(columns=COLUMNAS_REQUERIDAS)

    cliente = openai.OpenAI(api_key=api_key)
    estructura_prompt = estructura if estructura != "Indiferente" else "Instrumental o Vocal"
    prompt_experto = (
        f"Eres un DJ de música underground y crate digger. Actúa como una base "
        f"de datos. Necesito {cantidad_necesaria} canciones reales y "
        f"verificables de música electrónica (género: {genero}, estructura: "
        f"{estructura_prompt}, nivel de energía de 1-10: {energia}). REGLAS "
        f"ESTRICTAS: 1) PROHIBIDO inventar canciones. Incluye solo pistas "
        f"reales y verificables. 2) PROHIBIDA la música comercial (EDM, "
        f"Mainstage o Radio). Excluye siempre al artista '{artista_excluir}'. "
        f"3) Busca preferentemente en catálogos de sellos de culto como "
        f"Afterlife, Keinemusik, Drumcode, KNTXT, Innervisions, Solid Grooves, "
        f"Tresor, Kompakt, Defected o similares. 4) El BPM DEBE estar entre "
        f"{int(bpm_orig) - 3} y {int(bpm_orig) + 3}. 5) La clave armónica "
        f"DEBE ser compatible con {key_orig} según la Rueda Camelot. Devuelve "
        "ÚNICAMENTE un objeto JSON válido con la clave 'canciones', cuyo valor "
        "sea un array de objetos con las claves exactas: titulo, artista, "
        "genero, bpm (int), key_camelot, energia (int), estructura, sello. "
        "Ejemplo de JSON válido: "
        '{{"canciones": [{{"titulo": "The Age of Love (Charlotte de Witte '
        'Remix)", "artista": "Age of Love", "genero": "Techno", "bpm": 130, '
        '"key_camelot": "8A", "energia": 8, "estructura": "Instrumental", '
        '"sello": "KNTXT"}}]}}. No escribas markdown ni texto fuera del JSON.'
    )

    try:
        respuesta = cliente.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responde únicamente con un objeto JSON válido que "
                        "contenga la clave 'canciones'."
                    ),
                },
                {"role": "user", "content": prompt_experto},
            ],
        )
    except openai.OpenAIError as error:
        raise ValueError(f"No se pudo consultar OpenAI: {error}") from error
    contenido = respuesta.choices[0].message.content
    if not contenido:
        raise ValueError("OpenAI devolvió una respuesta vacía.")

    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as error:
        raise ValueError("OpenAI no devolvió un JSON válido.") from error

    if isinstance(datos, dict):
        datos = datos.get("canciones", datos.get("tracks", datos))
    if not isinstance(datos, list):
        raise ValueError("La respuesta de OpenAI no contiene un array de canciones.")

    filas = []
    minimo_bpm = int(bpm_orig) - 3
    maximo_bpm = int(bpm_orig) + 3
    for cancion in datos:
        if not isinstance(cancion, dict):
            continue
        if set(cancion) != set(COLUMNAS_REQUERIDAS):
            continue
        try:
            fila = {
                "titulo": str(cancion["titulo"]).strip(),
                "artista": str(cancion["artista"]).strip(),
                "genero": str(cancion["genero"]).strip(),
                "bpm": int(cancion["bpm"]),
                "key_camelot": str(cancion["key_camelot"]).strip().upper(),
                "energia": int(cancion["energia"]),
                "estructura": str(cancion["estructura"]).strip(),
                "sello": str(cancion["sello"]).strip(),
            }
        except (TypeError, ValueError):
            continue
        if (
            not fila["titulo"]
            or not fila["artista"]
            or fila["genero"] != genero
            or _artista_excluido(fila["artista"], artista_excluir)
            or not minimo_bpm <= fila["bpm"] <= maximo_bpm
            or not 1 <= fila["energia"] <= 10
            or fila["estructura"] not in {"Instrumental", "Vocal"}
            or not es_clave_compatible(key_orig, fila["key_camelot"])
            or (estructura != "Indiferente" and fila["estructura"] != estructura)
        ):
            continue
        filas.append(fila)
        if len(filas) == cantidad_necesaria:
            break

    dataframe_nuevo = pd.DataFrame(filas, columns=COLUMNAS_REQUERIDAS)
    if dataframe_nuevo.empty:
        return dataframe_nuevo

    dataset_actual = cargar_datos()
    dataset_actual = pd.concat(
        [dataset_actual, dataframe_nuevo],
        ignore_index=True,
    ).drop_duplicates(subset=["titulo", "artista"], keep="last")
    guardar_datos(dataset_actual)
    return dataframe_nuevo


def analizar_referencia(api_key, cancion, artista):
    """Obtiene el BPM y la clave Camelot de una canción mediante OpenAI."""
    if not api_key:
        raise ValueError(
            "Introduce la clave de OpenAI para poder analizar la canción."
        )

    cliente = openai.OpenAI(api_key=api_key)
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
    except openai.OpenAIError as error:
        raise ValueError(f"No se pudo analizar la referencia con OpenAI: {error}") from error

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
    bpm_orig=None,
    key_orig=None,
):
    """Devuelve hasta tres canciones compatibles con el patrón introducido."""
    if bpm_orig is None or key_orig is None:
        bpm_orig, key_orig = analizar_referencia(
            api_key,
            cancion,
            artista_usuario,
        )
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

    resultados_locales = dataframe.sort_values(
        by="distancia_energia",
        ascending=True,
    ).head(3)
    if len(resultados_locales) >= 3 or not api_key:
        return resultados_locales

    cantidad_necesaria = 3 - len(resultados_locales)
    resultados_nube = buscar_en_nube(
        api_key=api_key,
        artista_excluir=artista_usuario,
        genero=genero,
        energia=energia,
        estructura=estructura,
        bpm_orig=bpm_orig,
        key_orig=key_orig,
        cantidad_necesaria=cantidad_necesaria,
    )
    if resultados_nube.empty:
        return resultados_locales

    resultados_nube["distancia_energia"] = (
        resultados_nube["energia"] - int(energia)
    ).abs()
    return pd.concat(
        [resultados_locales, resultados_nube],
        ignore_index=True,
    ).head(3)
