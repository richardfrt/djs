"""Interfaz Streamlit de Underground Matcher Local."""

import io
import hashlib
from urllib.parse import quote_plus

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from database import COLUMNAS_REQUERIDAS, cargar_datos, guardar_datos
from ml_model import recomendar_matches


st.set_page_config(
    page_title="Underground Matcher Local",
    page_icon="🎚️",
    layout="wide",
)


GENEROS = ["Indiferente", "Techno", "Tech House", "Deep House", "Melodic"]
ESTRUCTURAS = ["Indiferente", "Instrumental", "Vocal"]
CLAVES_CAMELOT = [
    f"{numero}{letra}"
    for numero in range(1, 13)
    for letra in ("A", "B")
]


@st.cache_data(show_spinner=False)
def analizar_audio(archivo_subido):
    """Estima el BPM de un archivo de audio a partir de su beat tracking."""
    if not archivo_subido:
        raise ValueError("El archivo de audio está vacío.")

    audio, frecuencia_muestreo = librosa.load(
        io.BytesIO(archivo_subido),
        sr=None,
        mono=True,
    )
    if audio.size == 0:
        raise ValueError("El archivo de audio no contiene muestras.")

    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=frecuencia_muestreo,
    )
    valores_tempo = np.asarray(tempo).reshape(-1)
    if valores_tempo.size == 0 or not np.isfinite(valores_tempo[0]):
        raise ValueError("No se pudo estimar un BPM válido.")

    bpm_detectado = int(round(float(valores_tempo[0])))
    if not 40 <= bpm_detectado <= 220:
        raise ValueError(
            f"El BPM detectado ({bpm_detectado}) está fuera del rango permitido."
        )
    return bpm_detectado


def url_youtube(titulo, artista):
    consulta = quote_plus(f"{artista} {titulo}")
    return f"https://www.youtube.com/results?search_query={consulta}"


def procesar_csv_subido(archivo):
    """Concatena el CSV subido, elimina duplicados y persiste el dataset."""
    nuevo_dataframe = pd.read_csv(archivo)
    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in nuevo_dataframe.columns
    ]
    if faltantes:
        raise ValueError(
            "El CSV no contiene las columnas requeridas: "
            + ", ".join(faltantes)
        )

    nuevo_dataframe = nuevo_dataframe[COLUMNAS_REQUERIDAS].copy()
    nuevo_dataframe["bpm"] = pd.to_numeric(
        nuevo_dataframe["bpm"],
        errors="raise",
    ).astype(int)
    nuevo_dataframe["energia"] = pd.to_numeric(
        nuevo_dataframe["energia"],
        errors="raise",
    ).astype(int)

    dataset_actual = cargar_datos()
    dataset_completo = pd.concat(
        [dataset_actual, nuevo_dataframe],
        ignore_index=True,
    )
    dataset_completo = dataset_completo.drop_duplicates(
        subset=["titulo", "artista"],
        keep="last",
    )
    guardar_datos(dataset_completo)
    return len(nuevo_dataframe), len(dataset_completo)


st.title("Underground Matcher Local 🎚️")
st.caption(
    "Harmonic Mixing estricto con BPM, energía y compatibilidad Camelot. "
    "Todo el procesamiento se ejecuta localmente."
)


with st.sidebar:
    st.header("☁️ Sistema híbrido")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Opcional: solo se usa si el dataset local devuelve menos de 3 resultados.",
    )
    st.caption(
        "Primero se buscan matches en dataset.csv. La nube solo completa los "
        "resultados que falten y los guarda en el dataset local."
    )

    st.divider()
    st.header("📥 Carga masiva de entrenamiento")
    st.write(
        "El CSV debe incluir: titulo, artista, genero, bpm, "
        "key_camelot, energia, estructura y sello."
    )
    archivo_csv = st.file_uploader(
        "Subir canciones en CSV",
        type=["csv"],
    )

    if archivo_csv is not None:
        identificador_archivo = (
            f"{archivo_csv.name}:{archivo_csv.size}"
        )
        if st.session_state.get("archivo_procesado") != identificador_archivo:
            try:
                filas_recibidas, total_dataset = procesar_csv_subido(
                    archivo_csv
                )
                st.session_state["archivo_procesado"] = identificador_archivo
                st.success(
                    f"CSV procesado: {filas_recibidas} filas recibidas. "
                    f"Dataset actual: {total_dataset} canciones."
                )
            except (ValueError, pd.errors.ParserError) as error:
                st.error(f"No se pudo procesar el CSV: {error}")


columna_izquierda, columna_derecha = st.columns(2)

with columna_izquierda:
    st.subheader("Pista de referencia")
    cancion_referencia = st.text_input(
        "Canción (referencia)",
        placeholder="Ej. cualquier canción del mundo",
    )
    artista_original = st.text_input(
        "Artista original",
        placeholder="Obligatorio para excluirlo",
    )
    archivo_audio = st.file_uploader(
        "🎵 Arrastra tu track para auto-detectar BPM (MP3/WAV)",
        type=["mp3", "wav"],
        help="El análisis se ejecuta localmente y no sube el audio a ningún servicio.",
    )

    if "bpm_original" not in st.session_state:
        st.session_state["bpm_original"] = 126

    if archivo_audio is not None:
        bytes_audio = archivo_audio.getvalue()
        huella_audio = hashlib.sha256(bytes_audio).hexdigest()
        if st.session_state.get("audio_analizado") != huella_audio:
            try:
                with st.spinner("Analizando ondas de sonido..."):
                    bpm_detectado = analizar_audio(bytes_audio)
                st.session_state["bpm_original"] = bpm_detectado
                st.session_state["audio_analizado"] = huella_audio
                st.success(f"BPM detectado automáticamente: {bpm_detectado}")
            except (ValueError, OSError, RuntimeError, sf.SoundFileError) as error:
                st.error(f"No se pudo analizar el audio: {error}")

    bpm_original = st.number_input(
        "BPM de la pista",
        min_value=40,
        max_value=220,
        step=1,
        key="bpm_original",
    )
    key_original = st.selectbox(
        "Clave armónica (Camelot)",
        CLAVES_CAMELOT,
        index=CLAVES_CAMELOT.index("8A"),
    )

with columna_derecha:
    st.subheader("Criterios de búsqueda")
    genero_buscado = st.selectbox(
        "Género buscado",
        GENEROS,
    )
    estructura_buscada = st.selectbox(
        "Estructura",
        ESTRUCTURAS,
    )
    energia_deseada = st.slider(
        "Energía deseada",
        min_value=1,
        max_value=10,
        value=7,
    )


if st.button(
    "Encontrar Matches Perfectos",
    type="primary",
    use_container_width=True,
):
    if not artista_original.strip():
        st.warning("Introduce el artista original para continuar.")
    else:
        try:
            resultados = recomendar_matches(
                artista_usuario=artista_original,
                genero=genero_buscado,
                energia=energia_deseada,
                estructura=estructura_buscada,
                bpm_orig=bpm_original,
                key_orig=key_original,
                api_key=openai_api_key.strip() or None,
            )
            st.session_state["resultados"] = resultados
            st.session_state["referencia"] = cancion_referencia.strip()
            st.session_state["artista_referencia"] = artista_original.strip()
        except ValueError as error:
            st.error(str(error))


if "resultados" in st.session_state:
    resultados = st.session_state["resultados"]

    if resultados.empty:
        st.error(
            "No hay canciones en el dataset que cumplan simultáneamente "
            "los filtros de BPM, género, estructura y compatibilidad Camelot."
        )
    else:
        st.subheader("Matches perfectos")
        st.caption(
            f"Referencia: {st.session_state['referencia'] or 'Sin título'} · "
            f"Artista excluido: {st.session_state['artista_referencia']}"
        )

        for _, cancion in resultados.iterrows():
            col_pista, col_bpm, col_key = st.columns([3, 1, 1])
            with col_pista:
                st.write(
                    f"### {cancion['titulo']}"
                )
                st.write(
                    f"**{cancion['artista']}** · "
                    f"Sello: {cancion['sello']}"
                )
                st.link_button(
                    "▶ Buscar en YouTube",
                    url_youtube(
                        cancion["titulo"],
                        cancion["artista"],
                    ),
                )
            with col_bpm:
                st.metric("BPM", int(cancion["bpm"]))
            with col_key:
                st.metric("Key", cancion["key_camelot"])
            st.divider()
