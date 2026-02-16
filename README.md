---

# Llama-3.2-1B-Espanol: Crónica de una Reeducación Conceptual

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffd21e)](https://huggingface.co/CEGTEdicion/Llama-3.2-1B-espanol)
Este repositorio documenta el proceso de especialización del modelo **Llama 3.2-1B**, transformando un modelo de parámetros reducidos en un asistente académico capaz de disertar con precisión sobre ciencias biológicas y fotosíntesis en un español fluido y natural.

## 📝 Filosofía del Proyecto

En la era de los Grandes Modelos de Lenguaje (LLM), los modelos pequeños suelen enfrentar el desafío de las **alucinaciones técnicas** debido a su capacidad limitada de almacenamiento de hechos. Este proyecto nace de una premisa académica: *no se trata de cuántos parámetros tiene el modelo, sino de la calidad de la "dieta" informativa con la que se entrena*. Mediante la **Destilación de Datos Sintéticos** y el uso de **LoRA**, refinamos el razonamiento del modelo sin sacrificar su agilidad.

---

## 🛠️ El Ciclo de Desarrollo (Arquitectura de Scripts)

El flujo de trabajo se ha diseñado como una secuencia lógica de cuatro actos:

### 01. Preparación del Laboratorio (`01_setup.sh`)

La base de cualquier investigación de IA es un entorno predecible. Este script no solo instala dependencias, sino que garantiza que la comunicación entre la GPU y **Unsloth** sea óptima.

* **Recomendación:** Siempre verifica que tu sesión tenga asignada una GPU (T4 o superior). El éxito de este script depende de que las librerías `xformers` y `bitsandbytes` se alineen correctamente con la arquitectura CUDA del sistema.

### 02. Neuroplasticidad Controlada (`02_load_model.py`)

Aquí cargamos la estructura cerebral del modelo. Aplicamos **cuantización de 4 bits** para que el modelo sea ligero, pero activamos los adaptadores LoRA en todos los módulos clave (`q, k, v, o, gate, up, down_proj`).

* **Recomendación:** Al entrenar modelos de 1B para conceptos científicos, es preferible usar un rango (`r`) de 16 o 32. Rangos menores pueden ser insuficientes para capturar la terminología técnica compleja.

### 03. El Proceso de Reeducación (`03_train_model.py`)

El motor de aprendizaje. Este script es donde inyectamos el conocimiento corregido. El modelo aprende a sustituir sus errores previos por verdades científicas.
Dataset recomendados para practicar como: bertin-project/alpaca-spanish, plncmm/spanish-alpaca, saillab/alpaca-spanish-cleaned

* **Recomendación Crítica:** No satures el modelo con datos repetitivos. La clave es el **intercalado**: mezcla ensayos profundos generados por modelos mayores (Gemini) con preguntas cortas y precisas (Q&A). Esto evita que el modelo pierda su capacidad de síntesis mientras gana profundidad.

### 04. Inferencia y Sincronización del Conocimiento (`04_inference_and_export.py`)

El acto final donde validamos la "nueva conciencia" del modelo y la fusionamos en un archivo de 16 bits para su distribución global.

* **Recomendación:** Antes de la subida final a Hugging Face, realiza pruebas de "contraste". Hazle al modelo la misma pregunta que antes fallaba (ej: el uso del oxígeno). Si la respuesta es correcta y fluida, el modelo está listo para el mundo.

---

## 🔐 Seguridad y Gobernanza de Datos

La integridad de tu repositorio oficial `CEGTEdicion/Llama-3.2-1B-espanol` depende de una gestión de credenciales impecable.

* **Token de Acceso:** Utiliza siempre un token con permisos de **Escritura (Write)**.
* **Documentación de Secretos:** Nunca escribas tu API Key directamente en el código; utiliza el sistema de secretos de tu entorno (Colab Secrets o variables de entorno) para mantener tu "llave maestra" protegida.

---

## 🚀 Hoja de Ruta para el Usuario

1. **Clonación:** Trae el código a tu entorno local.
2. **Suministro de Datos:** Coloca tu archivo `tu_dataset.json` en la raíz. Asegúrate de que siga el formato Alpaca: `{"instruction": "...", "output": "..."}`.
3. **Ejecución:** Sigue el orden numérico de los scripts. Si cambias la lógica en el Script 03, recuerda que el Script 04 heredará esos cambios en la inferencia.

---

**Autor:** CEGTEdicion

**Visión:** Democratizar la inteligencia artificial de alta precisión, demostrando que el tamaño del modelo no es un límite para la excelencia académica.

---
