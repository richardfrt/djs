import hashlib
from io import StringIO

import streamlit as st

from database import cargar_csv_entrenamiento, guardar_nueva_cancion
from ml_model import recomendar_matches


st.set_page_config(page_title="Underground Matcher Local", page_icon="🎚️", layout="wide")

GENEROS = ["Techno", "Tech House", "Melodic"]
ESTRUCTURAS = ["Indiferente", "Instrumental", "Vocal"]


def crear_setlist_txt(canciones, perfil, patron):
    lineas = [
        "UNDERGROUND MATCHER LOCAL",
        "=========================",
        f"Referencia: {perfil['cancion']} — {perfil['artista']}",
        f"Perfil: {perfil['genero']} | Energía: {perfil['energia']}/10 | Estructura: {perfil['estructura']}",
        f"Patrón estimado: {patron['bpm']} BPM | Camelot: {patron['key_camelot']}",
        "",
    ]
    for posicion, (_, cancion) in enumerate(canciones.iterrows(), 1):
        lineas.extend(
            [
                f"{posicion}. {cancion['titulo']} — {cancion['artista']}",
                f"   {cancion['genero']} | {cancion['bpm']} BPM | Camelot {cancion['key_camelot']}",
                f"   Estructura: {cancion['estructura']} | Sello: {cancion['sello']}",
                f"   YouTube: {cancion['youtube_url']}",
                "",
            ]
        )
    return "\n".join(lineas)


def render_tarjeta(cancion, posicion):
    with st.container(border=True):
        st.caption(f"RECOMENDACIÓN {posicion}")
        st.subheader(cancion["titulo"])
        st.write(f"**{cancion['artista']}** · {cancion['genero']}")
        metricas = st.columns(2)
        metricas[0].metric("BPM", int(cancion["bpm"]))
        metricas[1].metric("Camelot", cancion["key_camelot"])
        st.write(f"**Energía:** {int(cancion['energia'])}/10 · **{cancion['estructura']}**")
        st.write(
            f"**Sello:** {cancion['sello']} · "
            f"{'Independiente' if cancion['sello_independiente'] else 'No independiente'}"
        )
        st.caption(" · ".join(cancion["tags_match"]))
        st.caption(cancion["compatibilidad"])
        st.link_button("▶ Buscar en YouTube", cancion["youtube_url"], use_container_width=True)


st.markdown(
    """
    <style>
        .block-container { max-width: 1180px; padding-top: 2rem; }
        .eyebrow { color: #a78bfa; font-size: .78rem; font-weight: 700; letter-spacing: .16em; }
        .hero h1 { font-size: 2.9rem; margin: .2rem 0; }
        .hero p { color: #9aa7b8; font-size: 1.05rem; }
    </style>
    <div class="hero">
        <div class="eyebrow">100% LOCAL · DATASET EXTENSIBLE · SIN APIs</div>
        <h1>Underground Matcher Local 🎚️</h1>
        <p>Recomendaciones por patrones de energía, BPM, armonía y etiquetas musicales.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("🎓 Enseñar al sistema")
    st.caption("Las canciones nuevas quedan disponibles en memoria para futuras búsquedas del proceso actual.")
    with st.form("entrenamiento"):
        nuevo_titulo = st.text_input("Título")
        nuevo_artista = st.text_input("Artista")
        nuevo_genero = st.selectbox("Género", GENEROS, key="nuevo_genero")
        nuevo_bpm = st.number_input("BPM", min_value=80, max_value=180, value=124)
        nueva_clave = st.text_input("Clave Camelot", value="8A")
        nueva_energia = st.slider("Energía", 1, 10, 6, key="nueva_energia")
        nueva_estructura = st.selectbox("Estructura", ["Instrumental", "Vocal"], key="nueva_estructura")
        nuevo_sello = st.text_input("Sello independiente")
        nuevos_tags = st.text_input("Tags", placeholder="deep, hipnotico, groove")
        entrenar = st.form_submit_button("Entrenar / Añadir al Dataset", use_container_width=True)

    if entrenar:
        try:
            cancion_anadida = guardar_nueva_cancion(
                {
                    "titulo": nuevo_titulo,
                    "artista": nuevo_artista,
                    "genero": nuevo_genero,
                    "bpm": nuevo_bpm,
                    "key_camelot": nueva_clave,
                    "energia": nueva_energia,
                    "estructura": nueva_estructura,
                    "sello_independiente": True,
                    "sello": nuevo_sello,
                    "tags_match": nuevos_tags.split(","),
                }
            )
            st.success(f"Añadida: {cancion_anadida['titulo']}")
        except (TypeError, ValueError) as error:
            st.error(str(error))

    st.divider()
    st.header("📥 Carga Masiva de Entrenamiento")
    st.caption(
        "Sube cientos de canciones en un CSV. Las filas nuevas se incorporan "
        "al dataset local y estarán disponibles inmediatamente."
    )
    plantilla_csv = StringIO()
    plantilla_csv.write(
        "titulo,artista,genero,bpm,key_camelot,energia,estructura,sello\n"
        "Ejemplo Track,Ejemplo Artist,Techno,128,8A,7,Instrumental,Sello Local\n"
    )
    st.download_button(
        "⬇ Descargar plantilla CSV",
        data=plantilla_csv.getvalue(),
        file_name="plantilla_entrenamiento.csv",
        mime="text/csv",
        use_container_width=True,
    )
    archivo_csv = st.file_uploader(
        "Selecciona tu archivo de entrenamiento",
        type=["csv"],
        help=(
            "Columnas obligatorias: titulo, artista, genero, bpm, "
            "key_camelot, energia y estructura."
        ),
    )

    if archivo_csv is not None:
        contenido_csv = archivo_csv.getvalue()
        huella_csv = hashlib.sha256(contenido_csv).hexdigest()
        if st.session_state.get("csv_entrenado") != huella_csv:
            try:
                canciones_anadidas = cargar_csv_entrenamiento(StringIO(contenido_csv.decode("utf-8-sig")))
                st.session_state["csv_entrenado"] = huella_csv
                if canciones_anadidas:
                    st.success(
                        f"✅ Carga completada: {canciones_anadidas} canciones añadidas."
                    )
                else:
                    st.info("El CSV no contenía canciones nuevas.")
            except (UnicodeDecodeError, ValueError) as error:
                st.error(f"No se pudo cargar el CSV: {error}")


with st.form("busqueda"):
    st.subheader("Define tu punto de partida")
    columna_1, columna_2 = st.columns(2)
    with columna_1:
        cancion_input = st.text_input("Canción de partida", placeholder="Ej. Midnight Driver")
    with columna_2:
        artista_input = st.text_input("Artista de partida", placeholder="Ej. Artista de referencia")

    columna_3, columna_4, columna_5 = st.columns([1.2, 1, 1])
    with columna_3:
        genero = st.selectbox("Género de la sesión", GENEROS)
    with columna_4:
        energia = st.slider("Energía deseada", 1, 10, 6)
    with columna_5:
        estructura = st.selectbox("Estructura", ESTRUCTURAS)
    buscar = st.form_submit_button("Calcular Coincidencias Underground", type="primary", use_container_width=True)


if buscar:
    resultados, patron = recomendar_matches(
        cancion_input, artista_input, genero, energia, estructura
    )
    st.session_state["resultados"] = resultados
    st.session_state["patron"] = patron
    st.session_state["perfil"] = {
        "cancion": cancion_input.strip() or "Canción de referencia",
        "artista": artista_input.strip() or "Artista no especificado",
        "genero": genero,
        "energia": energia,
        "estructura": estructura,
    }


if "resultados" in st.session_state:
    resultados = st.session_state["resultados"]
    perfil = st.session_state["perfil"]
    patron = st.session_state["patron"]
    st.divider()
    st.subheader(f"Coincidencias para “{perfil['cancion']}”")
    st.caption(
        f"{perfil['artista']} · {perfil['genero']} · energía {perfil['energia']}/10 · "
        f"patrón {patron['bpm']} BPM / {patron['key_camelot']}"
    )
    if resultados.empty:
        st.warning("No hay canciones compatibles con los filtros seleccionados.")
    else:
        resumen = st.columns(3)
        resumen[0].metric("Matches", len(resultados))
        resumen[1].metric("Energía objetivo", f"{perfil['energia']}/10")
        resumen[2].metric("Clave estimada", patron["key_camelot"])
        tarjetas = st.columns(3)
        for posicion, (columna, (_, cancion)) in enumerate(zip(tarjetas, resultados.iterrows()), 1):
            with columna:
                render_tarjeta(cancion, posicion)
        st.download_button(
            "⬇ Descargar setlist (.txt)",
            crear_setlist_txt(resultados, perfil, patron),
            "underground_matcher_setlist.txt",
            "text/plain",
            use_container_width=True,
        )
else:
    st.info("Completa el perfil y pulsa el botón para calcular tus coincidencias.")
