"""Dataset local y extensible para Underground Matcher Local.

El almacenamiento es deliberadamente local: ``guardar_nueva_cancion`` actualiza
la lista en memoria del proceso de Streamlit, sin APIs ni servicios externos.
"""

from copy import deepcopy

import pandas as pd


COLUMNAS_DATASET = [
    "id",
    "titulo",
    "artista",
    "genero",
    "bpm",
    "key_camelot",
    "energia",
    "estructura",
    "sello_independiente",
    "sello",
    "tags_match",
]


CANCIONES = [
    [1, "Spastik", "Plastikman", "Techno", 130, "8A", 9, "Instrumental", True, ["hipnotico", "industrial", "percusion"]],
    [2, "The Bells", "Jeff Mills", "Techno", 131, "9A", 10, "Instrumental", True, ["peak-time", "acid", "driving"]],
    [3, "Knights of the Jaguar", "The Aztec Mystic", "Techno", 128, "7A", 8, "Instrumental", True, ["detroit", "trance", "melodico"]],
    [4, "Losing Control", "Laurent Garnier", "Techno", 127, "8B", 8, "Vocal", True, ["vocal", "rave", "hipnotico"]],
    [5, "La Real", "Antenes", "Techno", 126, "10A", 7, "Vocal", True, ["underground", "groove", "vocal"]],
    [6, "Your Mind", "Dax J", "Techno", 140, "11A", 10, "Vocal", True, ["hard", "rave", "peak-time"]],
    [7, "The Age of Love (Charlotte de Witte & Enrico Sangiuliano Remix)", "Age of Love", "Techno", 132, "2A", 9, "Vocal", True, ["trance", "rave", "vocal"]],
    [8, "Lola's Theme", "The Shapeshifters", "Tech House", 125, "5A", 7, "Vocal", True, ["vocal", "groove", "disco"]],
    [9, "Cola", "CamelPhat & Elderbrook", "Tech House", 123, "8A", 7, "Vocal", True, ["vocal", "bass", "groove"]],
    [10, "The Creeps (Get on the Dancefloor)", "Freaks", "Tech House", 126, "6A", 8, "Vocal", True, ["vocal", "funky", "groove"]],
    [11, "Hungry for the Power", "azari & III", "Tech House", 124, "9B", 8, "Vocal", True, ["vocal", "indie-dance", "bass"]],
    [12, "The Shining Path", "Joris Voorn", "Tech House", 124, "4B", 6, "Instrumental", True, ["melodico", "groove", "deep"]],
    [13, "Deep Burnt", "Pépé Bradock", "Deep House", 120, "6A", 5, "Instrumental", True, ["deep", "jazzy", "warm"]],
    [14, "At Night", "Shakedown", "Deep House", 124, "8A", 7, "Vocal", True, ["vocal", "disco", "classic"]],
    [15, "Finally", "Kings of Tomorrow", "Deep House", 122, "5B", 6, "Vocal", True, ["vocal", "soulful", "warm"]],
    [16, "Deep Inside", "Hardrive", "Deep House", 124, "7A", 7, "Vocal", True, ["vocal", "piano", "classic"]],
    [17, "You Got the Love", "Candi Staton", "Deep House", 123, "9B", 8, "Vocal", True, ["vocal", "soulful", "anthemic"]],
    [18, "Opus", "Eric Prydz", "Melodic", 126, "8B", 8, "Instrumental", True, ["melodico", "progressive", "euforico"]],
    [19, "Atlas", "Adriatique", "Melodic", 122, "9B", 6, "Instrumental", True, ["melodico", "atmosferico", "deep"]],
    [20, "The Last Dancer", "Khen", "Melodic", 120, "6B", 5, "Instrumental", True, ["melodico", "organico", "deep"]],
    [21, "You Are My Destiny", "Rodriguez Jr.", "Melodic", 123, "10B", 7, "Vocal", True, ["vocal", "melodico", "warm"]],
    [22, "Eternity", "Stephan Bodzin", "Melodic", 124, "7B", 8, "Instrumental", True, ["melodico", "analogico", "hypnotic"]],
]

SELLOS_POR_ID = {
    1: "Plus 8 Records", 2: "Purpose Maker", 3: "Interdimensional Transmissions",
    4: "F-Communications", 5: "Mental Disorder", 6: "Monolith Records",
    7: "Afterlife", 8: "Positiva Records", 9: "Defected Records",
    10: "Playhouse", 11: "Modular Recordings", 12: "Green",
    13: "Atavisme", 14: "Defected Records", 15: "Distance Music",
    16: "Nervous Records", 17: "Relief Records", 18: "Pryda Recordings",
    19: "Afterlife", 20: "Lost & Found", 21: "Mobilee Records",
    22: "Herzblut Recordings",
}


def obtener_dataset():
    """Devuelve una copia independiente del dataset actual."""
    columnas_base = [columna for columna in COLUMNAS_DATASET if columna != "sello"]
    dataset = pd.DataFrame(CANCIONES, columns=columnas_base).copy(deep=True)
    dataset["sello"] = dataset["id"].map(SELLOS_POR_ID).fillna("Sello independiente")
    return dataset[COLUMNAS_DATASET]


def guardar_nueva_cancion(nueva_fila):
    """Valida y añade una canción al dataset local en memoria.

    ``nueva_fila`` puede ser un diccionario o una única fila de pandas.
    Devuelve la canción normalizada que fue incorporada.
    """
    if isinstance(nueva_fila, pd.Series):
        nueva_fila = nueva_fila.to_dict()
    if not isinstance(nueva_fila, dict):
        raise TypeError("nueva_fila debe ser un diccionario o una Serie de pandas.")

    obligatorias = {"titulo", "artista", "genero", "bpm", "key_camelot", "energia", "estructura", "sello_independiente", "tags_match"}
    faltantes = obligatorias - set(nueva_fila)
    if faltantes:
        raise ValueError(f"Faltan campos obligatorios: {', '.join(sorted(faltantes))}")

    titulo = str(nueva_fila["titulo"]).strip()
    artista = str(nueva_fila["artista"]).strip()
    if not titulo or not artista:
        raise ValueError("El título y el artista no pueden estar vacíos.")

    genero = str(nueva_fila["genero"]).strip()
    estructura = str(nueva_fila["estructura"]).strip()
    if genero not in {"Techno", "Tech House", "Melodic"}:
        raise ValueError("El género debe ser Techno, Tech House o Melodic.")
    if estructura not in {"Instrumental", "Vocal"}:
        raise ValueError("La estructura debe ser Instrumental o Vocal.")

    bpm = int(nueva_fila["bpm"])
    energia = int(nueva_fila["energia"])
    if not 80 <= bpm <= 180:
        raise ValueError("El BPM debe estar entre 80 y 180.")
    if not 1 <= energia <= 10:
        raise ValueError("La energía debe estar entre 1 y 10.")

    clave = str(nueva_fila["key_camelot"]).strip().upper()
    if len(clave) < 2 or clave[:-1].isdigit() is False or clave[-1] not in {"A", "B"} or not 1 <= int(clave[:-1]) <= 12:
        raise ValueError("La clave Camelot debe tener el formato 1A-12B.")

    nueva = {
        "id": max((fila[0] for fila in CANCIONES), default=0) + 1,
        "titulo": titulo,
        "artista": artista,
        "genero": genero,
        "bpm": bpm,
        "key_camelot": clave,
        "energia": energia,
        "estructura": estructura,
        "sello_independiente": bool(nueva_fila["sello_independiente"]),
        "tags_match": [str(tag).strip().lower() for tag in nueva_fila["tags_match"] if str(tag).strip()],
    }
    CANCIONES.append([nueva[columna] for columna in COLUMNAS_DATASET if columna != "sello"])
    SELLOS_POR_ID[nueva["id"]] = str(nueva_fila.get("sello") or "Sello independiente").strip()
    return deepcopy(nueva)
