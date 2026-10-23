# 🧠 Detección de Depresión mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-NLP%20%2F%20Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositorio contiene los modelos, experimentos y metodologías orientados a la detección temprana de signos de depresión a partir del corpus **MentalRiskES (ediciones 2023 y 2024)**.

---

## 📌 Descripción del Proyecto

El objetivo principal de este proyecto es documentar, almacenar y evaluar experimentos de entrenamiento y desarrollo de modelos de Machine Learning aplicados a la salud mental, como parte de mi trabajo de investigación. 

Además, busca:
- 🔬 **Facilitar la reproducibilidad** de los experimentos realizados.
- 📚 **Servir como referencia** para investigadoras e investigadores interesados en la detección automatizada de depresión y procesamiento de lenguaje natural (NLP) en español.

---

## 📊 Datos y Corpus

> **Nota:** Por motivos de privacidad y derechos de uso, los datos utilizados para entrenar y evaluar los modelos **no están incluidos** en este repositorio.

Para obtener acceso al corpus de entrenamiento y evaluación:
1. Visita la página oficial de los organizadores de [MentalRiskES](https://mentalriskes.github.io/). 
2. Sigue las indicaciones y requisitos establecidos por la organización para solicitar el acceso.

---

## ⚙️ Guía de Configuración

El proyecto gestiona la ubicación de los datos mediante variables de entorno locales definidas en un archivo `.env`.

### Pasos para configurar el entorno:

1. **Revisar la plantilla:** Consulta el archivo `ENV_ejemplo`, el cual contiene la estructura requerida de las variables de entorno.
2. **Crear el archivo `.env`:** Crea un archivo llamado `.env` en la raíz del proyecto.
3. **Definir las rutas:** Copia el contenido de `ENV_ejemplo` en tu nuevo archivo `.env` y reemplaza los valores predeterminados con las rutas absolutas/locales donde descargaste los corpus de MentalRiskES (2023 y 2024).
4. **Verificar estructura:** Asegúrate de que las rutas especificadas coincidan con la estructura de directorios de tu equipo.
   
---

## 🔁 Reproducibilidad

Para replicar exitosamente los experimentos documentados en este repositorio:
1. Obtén la aprobación y descarga de los corpus oficiales de **MentalRiskES**.
2. Configura las rutas correspondientes en tu archivo `.env` siguiendo las instrucciones anteriores.
3. Ejecuta los scripts o cuadernos de entrenamiento/evaluación correspondientes.
