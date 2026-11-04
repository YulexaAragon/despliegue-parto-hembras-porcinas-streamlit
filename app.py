# -*- coding: utf-8 -*-
"""
Predicción de Parto Exitoso en Hembras Porcinas
Despliegue del modelo de clasificación (Random Forest) con Streamlit.

Proyecto de Minería de Datos (metodología CRISP-DM)
Maestría en Ciencia de Datos – Universidad Pontificia Bolivariana

Autores:
    Yulexa Andrea Aragón Lemus
    Yasmin Judith Florez Zabala
    John Jairo Sevilla Rodriguez
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predicción Parto Porcino", page_icon="🐷", layout="centered")


# ---------------------------------------------------------------------------
# Carga del modelo (se guarda en caché para no recargarlo en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(ruta="modelo-class-partos.pkl"):
    with open(ruta, "rb") as f:
        modelo, labelencoder, variables = pickle.load(f)
    return modelo, labelencoder, list(variables)


modelo, labelencoder, variables = cargar_modelo()

# Nombres legibles de las variables predictoras
NOMBRES = {
    "NUMERO_PARTO": "Número de parto",
    "PROMEDIO_VIVOS_ANTERIORES": "Promedio vivos anteriores",
    "TOTAL_MUERTOS_ANTERIORES": "Total muertos anteriores",
    "RETORNOS_ACUMULADOS": "Retornos acumulados",
}

# Perfiles reproductivos obtenidos con K-Means (k = 6), centroides en unidades originales
PERFILES = pd.DataFrame(
    [
        ["Experiencia intermedia y estable", 6, 14, 3, 0, 54.1],
        ["Jóvenes de alto rendimiento", 3, 17, 1, 0, 68.6],
        ["Jóvenes de baja productividad", 3, 8, 1, 0, 41.9],
        ["Alto desgaste reproductivo", 9, 13, 5, 0, 33.8],
        ["Jóvenes eficientes", 3, 13, 1, 0, 55.5],
        ["Con retornos previos", 5, 13, 3, 1, 45.8],
    ],
    columns=["Perfil", "NUMERO_PARTO", "PROMEDIO_VIVOS_ANTERIORES",
             "TOTAL_MUERTOS_ANTERIORES", "RETORNOS_ACUMULADOS", "Éxito %"],
)

# Rangos de las variables (los mismos de los controles) para normalizar distancias
RANGOS = {
    "NUMERO_PARTO": (2, 19),
    "PROMEDIO_VIVOS_ANTERIORES": (0.0, 31.0),
    "TOTAL_MUERTOS_ANTERIORES": (0, 35),
    "RETORNOS_ACUMULADOS": (0, 9),
}


def perfil_mas_cercano(registro: pd.Series) -> pd.Series:
    """Devuelve el perfil de K-Means más parecido (distancia euclidiana con escala min-max)."""
    def escalar(df):
        return pd.DataFrame({v: (df[v] - lo) / (hi - lo) for v, (lo, hi) in RANGOS.items()})

    centroides = escalar(PERFILES)
    punto = escalar(registro.to_frame().T).iloc[0]
    distancias = np.sqrt(((centroides - punto) ** 2).sum(axis=1))
    return PERFILES.loc[distancias.idxmin()]


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#2E86AB;margin-bottom:0'>🐷 Predicción de Parto Exitoso</h1>"
    "<p style='text-align:center;opacity:.75;font-size:1.1rem;margin-top:.3rem'>"
    "Hembras porcinas · Modelo de clasificación Random Forest</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Estima, **antes de la inseminación**, la probabilidad de que el próximo parto de una hembra sea "
    "exitoso (lechones nacidos vivos iguales o superiores a la mediana de su raza), a partir de su "
    "historial reproductivo."
)
st.divider()

# ---------------------------------------------------------------------------
# Entrada de datos
# ---------------------------------------------------------------------------
st.subheader("📋 Datos de la hembra")
col1, col2 = st.columns(2)
with col1:
    numero_parto = st.slider("🔢 Número de parto", 2, 19, 4, 1,
                             help="Número de parto actual de la hembra")
    total_muertos = st.slider("💀 Total muertos anteriores", 0, 35, 2, 1,
                              help="Total de lechones muertos en partos anteriores")
with col2:
    promedio_vivos = st.slider("🐽 Promedio vivos anteriores", 0.0, 31.0, 13.0, 0.1,
                               help="Promedio de lechones nacidos vivos en partos anteriores")
    retornos = st.slider("🔄 Retornos acumulados", 0, 9, 0, 1,
                         help="Número de inseminaciones fallidas acumuladas")

registro = pd.Series({
    "NUMERO_PARTO": numero_parto,
    "PROMEDIO_VIVOS_ANTERIORES": promedio_vivos,
    "TOTAL_MUERTOS_ANTERIORES": total_muertos,
    "RETORNOS_ACUMULADOS": retornos,
})
X = registro[variables].to_frame().T.astype(float)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
probas = modelo.predict_proba(X)[0]
clases = labelencoder.inverse_transform(modelo.classes_)
prob_exito = float(probas[list(clases).index("EXITOSO")])
es_exitoso = prob_exito >= 0.5

st.divider()
st.subheader("🎯 Resultado")

if es_exitoso:
    st.success(f"### ✅ Parto exitoso probable\nProbabilidad estimada de éxito: **{prob_exito:.0%}**")
else:
    st.error(f"### ⚠️ Riesgo de parto no exitoso\nProbabilidad estimada de éxito: **{prob_exito:.0%}**")
st.progress(prob_exito, text=f"Probabilidad de éxito: {prob_exito:.0%}")

# Perfil reproductivo más parecido
perfil = perfil_mas_cercano(registro)
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("**🧬 Perfil reproductivo más parecido (K-Means)**")
    st.markdown(f"#### {perfil['Perfil']}")
    st.caption(
        f"Centroide del perfil → parto: {perfil['NUMERO_PARTO']} · "
        f"vivos promedio: {perfil['PROMEDIO_VIVOS_ANTERIORES']} · "
        f"muertos anteriores: {perfil['TOTAL_MUERTOS_ANTERIORES']} · "
        f"retornos: {perfil['RETORNOS_ACUMULADOS']}"
    )
with c2:
    st.metric("Tasa histórica de éxito del perfil", f"{perfil['Éxito %']:.1f}%".replace(".", ","))

# ---------------------------------------------------------------------------
# Qué pesa en la decisión
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 ¿Qué variables pesan más en el modelo?")
importancias = (
    pd.DataFrame({"Variable": [NOMBRES[v] for v in variables],
                  "Importancia (%)": modelo.feature_importances_ * 100})
    .sort_values("Importancia (%)", ascending=False)
)
for _, fila in importancias.iterrows():
    st.progress(fila["Importancia (%)"] / 100,
                text=f"{fila['Variable']} · {fila['Importancia (%)']:.0f}%")
st.caption(
    "El promedio de lechones vivos en partos anteriores es el factor más influyente: "
    "el desempeño histórico de la hembra es la mejor señal de su próximo parto."
)

with st.expander("📚 Perfiles reproductivos identificados con K-Means"):
    tabla = PERFILES.rename(columns=NOMBRES).set_index("Perfil")
    st.dataframe(tabla, width='stretch')

with st.expander("ℹ️ Sobre el modelo"):
    st.markdown(
        """
- **Metodología:** CRISP-DM, con datos reales del sector agroindustrial (350.981 registros tras la limpieza).
- **Modelo:** Random Forest de 200 árboles (profundidad máxima 7), validado con validación cruzada de 10 folds.
- **Desempeño:** precisión 61,7 % · exactitud 62,3 % · F1-score 56,0 %, superando la línea base definida (precisión > 60 %).
- **Uso recomendado:** apoyo a la decisión. El éxito reproductivo también depende de factores que no están
  en los datos (genética del verraco, nutrición, sanidad), por lo que el resultado es una probabilidad, no una certeza.
        """
    )

st.divider()
st.caption(
    "Proyecto de Minería de Datos · Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana · "
    "Yulexa Andrea Aragón Lemus, Yasmin Judith Florez Zabala y John Jairo Sevilla Rodriguez."
)
