"""Interfaz Streamlit de Underground Matcher Local."""

from urllib.parse import quote_plus

import pandas as pd
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
    st.header("🧠 Cerebro IA (Agente Explorador)")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.caption(
        "Necesaria para analizar la pista de referencia y completar resultados "
        "cuando el dataset local no tenga tres coincidencias."
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
    elif not api_key.strip():
        st.error(
            "Necesitas introducir la API Key de OpenAI para que el sistema "
            "analice la pista original."
        )
    else:
        api_key = api_key.strip()
        try:
            resultados, bpm_orig, key_orig = recomendar_matches(
                cancion=cancion_referencia,
                artista_usuario=artista_original,
                genero=genero_buscado,
                energia=energia_deseada,
                estructura=estructura_buscada,
                api_key=api_key,
            )
            st.success(
                f"Análisis completado: {bpm_orig} BPM · clave {key_orig}"
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["resultados"] = resultados
            st.session_state["referencia"] = cancion_referencia.strip()
            st.session_state["artista_referencia"] = artista_original.strip()
            st.session_state["bpm_orig"] = bpm_orig
            st.session_state["key_orig"] = key_orig


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
        st.info(
            f"Análisis de tu pista: {st.session_state['referencia']} corre a "
            f"{st.session_state['bpm_orig']} BPM en clave "
            f"{st.session_state['key_orig']}"
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

        texto_setlist = ""
        for posicion, (_, cancion) in enumerate(resultados.iterrows(), start=1):
            texto_setlist += (
                f"{posicion}. {cancion['titulo']} - {cancion['artista']} | "
                f"{int(cancion['bpm'])} BPM | Key: {cancion['key_camelot']} | "
                f"Sello: {cancion['sello']}\n"
            )

        st.download_button(
            label="📥 Descargar Setlist (.txt)",
            data=texto_setlist,
            file_name="underground_setlist.txt",
            mime="text/plain",
            use_container_width=True,
        )
