"""Persistencia local del dataset de canciones de Underground Matcher."""

from pathlib import Path

import pandas as pd


CSV_FILE = "dataset.csv"

COLUMNAS_REQUERIDAS = [
    "titulo",
    "artista",
    "genero",
    "bpm",
    "key_camelot",
    "energia",
    "estructura",
    "sello",
]

VALORES_POR_DEFECTO = {
    "titulo": "Desconocido",
    "artista": "Desconocido",
    "genero": "Desconocido",
    "bpm": 126,
    "key_camelot": "8A",
    "energia": 7,
    "estructura": "Desconocido",
    "sello": "Desconocido",
}

DATOS_INICIALES = [
    {
        "titulo": "Spastik",
        "artista": "Plastikman",
        "genero": "Techno",
        "bpm": 130,
        "key_camelot": "8A",
        "energia": 9,
        "estructura": "Instrumental",
        "sello": "Plus 8 Records",
    },
    {
        "titulo": "Knights of the Jaguar",
        "artista": "The Aztec Mystic",
        "genero": "Techno",
        "bpm": 128,
        "key_camelot": "7A",
        "energia": 8,
        "estructura": "Instrumental",
        "sello": "Interdimensional Transmissions",
    },
    {
        "titulo": "The Shining Path",
        "artista": "Joris Voorn",
        "genero": "Tech House",
        "bpm": 124,
        "key_camelot": "4B",
        "energia": 6,
        "estructura": "Instrumental",
        "sello": "Green",
    },
    {
        "titulo": "Deep Burnt",
        "artista": "Pépé Bradock",
        "genero": "Deep House",
        "bpm": 120,
        "key_camelot": "6A",
        "energia": 5,
        "estructura": "Instrumental",
        "sello": "Atavisme",
    },
    {
        "titulo": "Atlas",
        "artista": "Adriatique",
        "genero": "Melodic",
        "bpm": 122,
        "key_camelot": "9B",
        "energia": 6,
        "estructura": "Instrumental",
        "sello": "Afterlife",
    },
]


def cargar_datos():
    """Carga el dataset persistente o crea el CSV inicial."""
    ruta = Path(CSV_FILE)
    if not ruta.exists():
        dataframe = pd.DataFrame(DATOS_INICIALES, columns=COLUMNAS_REQUERIDAS)
        guardar_datos(dataframe)
        return dataframe

    try:
        dataframe = pd.read_csv(ruta)
    except pd.errors.EmptyDataError:
        dataframe = pd.DataFrame()
    for columna in COLUMNAS_REQUERIDAS:
        if columna not in dataframe.columns:
            dataframe[columna] = VALORES_POR_DEFECTO[columna]

    dataframe = dataframe[COLUMNAS_REQUERIDAS].copy()
    dataframe["bpm"] = (
        pd.to_numeric(dataframe["bpm"], errors="coerce")
        .fillna(VALORES_POR_DEFECTO["bpm"])
        .astype(int)
    )
    dataframe["energia"] = pd.to_numeric(
        dataframe["energia"], errors="coerce"
    ).fillna(VALORES_POR_DEFECTO["energia"]).astype(int)
    for columna in ("titulo", "artista", "genero", "key_camelot", "estructura", "sello"):
        dataframe[columna] = dataframe[columna].fillna(
            VALORES_POR_DEFECTO[columna]
        )
    return dataframe


def guardar_datos(df):
    """Sobrescribe el dataset local sin guardar el índice de pandas."""
    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in df.columns
    ]
    if faltantes:
        raise ValueError(
            "No se puede guardar el dataset. Faltan columnas: "
            + ", ".join(faltantes)
        )

    dataframe = df[COLUMNAS_REQUERIDAS].copy()
    dataframe.to_csv(CSV_FILE, index=False)
