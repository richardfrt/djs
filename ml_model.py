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
    prompt = (
        f"Eres un DJ de música underground y crate digger. Actúa como una base "
        f"de datos. Necesito {cantidad_necesaria} canciones reales de música "
        f"electrónica (género: {genero}, estructura: {estructura_prompt}, nivel "
        f"de energía de 1-10: {energia}). REGLAS DE ORO: 1) El artista NO "
        f"puede ser {artista_excluir} ni nadie mainstream. 2) El BPM DEBE estar "
        f"entre {int(bpm_orig) - 3} y {int(bpm_orig) + 3}. 3) La clave armónica "
        f"DEBE ser compatible con {key_orig} según la Rueda Camelot. Devuelve "
        "ÚNICAMENTE un array JSON válido donde cada objeto tenga las claves "
        "exactas: titulo, artista, genero, bpm (int), key_camelot, energia "
        "(int), estructura, sello. No escribas markdown ni texto fuera del JSON."
    )

    try:
        respuesta = cliente.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Responde únicamente con un array JSON válido.",
                },
                {"role": "user", "content": prompt},
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


def recomendar_matches(
    artista_usuario,
    genero,
    energia,
    estructura,
    bpm_orig,
    key_orig,
    api_key=None,
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
