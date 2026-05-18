# Predicción de Parto Exitoso en Hembras Porcinas

Aplicación web desarrollada con Streamlit para predecir si el parto de una hembra porcina será exitoso o no, basada en su historial reproductivo.

## Descripción

Este proyecto forma parte de un trabajo de minería de datos con metodología CRISP-DM, desarrollado como parte de la Maestría en Ciencia de Datos. El modelo de clasificación fue entrenado con 350.981 registros de hembras porcinas y utiliza un Árbol de Decisión como modelo predictivo.

## Variable objetivo

`PARTO_EXITOSO`: un parto se clasifica como **EXITOSO** cuando el número de lechones nacidos vivos es mayor o igual a la mediana de su raza.

## Variables predictoras

| Variable | Descripción |
|---|---|
| `NUMERO_PARTO` | Número de parto actual de la hembra |
| `PROMEDIO_VIVOS_ANTERIORES` | Promedio de lechones nacidos vivos en partos anteriores |
| `TOTAL_MUERTOS_ANTERIORES` | Total de lechones muertos en partos anteriores |
| `RETORNOS_ACUMULADOS` | Número de inseminaciones fallidas acumuladas |

## Modelo

- **Algoritmo:** Árbol de Decisión
- **Accuracy:** 62.2%
- **Precisión:** 61.4%
- **F1-Score:** 56.3%

## Cómo ejecutar

1. Clona el repositorio:
```bash
git clone https://github.com/YulexaAragon/despliegue-parto-hembras-porcinas-streamlit.git
cd despliegue-parto-hembras-porcinas-streamlit
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```bash
streamlit run app.py
```

## Autores

- Yulexa Andrea Aragón Lemus
- Yasmin Judith Florez Zabala
- John Jairo Sevilla Rodriguez  
