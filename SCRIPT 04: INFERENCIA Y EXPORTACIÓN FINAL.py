# ==============================================================================
# SECCIÓN DE SEGURIDAD: AUTENTICACIÓN EN HUGGING FACE
# ==============================================================================
# Para subir modelos o descargar datasets privados, es necesario autenticarse.
# Existen dos métodos principales:
#
# MÉTODO A (Interactivo):
# from huggingface_hub import login
# login()  # Esto abrirá un cuadro de texto para pegar el Token (Write Access).
#
# MÉTODO B (Mediante Script):
# Si prefieres automatizarlo o renovar una clave obsoleta, usa:
# import os
# os.environ["HF_TOKEN"] = "tu_nuevo_token_aqui"
# ==============================================================================

from unsloth import FastLanguageModel
import torch
from huggingface_hub import login

# Si decides renovar el token manualmente, descomenta la siguiente línea:
# login(token="tu_token_aqui")

# 1. Preparación para Inferencia
# Activamos el modo nativo de Unsloth para una respuesta inmediata.
FastLanguageModel.for_inference(model) 

# 2. El Momento de la Verdad: Prueba de Precisión
# Usamos una pregunta que antes generaba errores para verificar la mejora.
instruccion_test = "¿Cuál es el papel del CO2 y qué gas liberan las plantas?"

inputs = tokenizer(
[
    alpaca_prompt.format(
        instruccion_test, 
        "",               
    )
], return_tensors = "pt").to("cuda")

print(f"--- Generando respuesta para: {instruccion_test} ---")
outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
respuesta = tokenizer.decode(outputs[0], skip_special_tokens = True)

# Limpiamos la salida para mostrar solo el bloque de la "Respuesta"
print("\n" + "="*30)
print("RESPUESTA DEL MODELO:")
print(respuesta.split("### Respuesta:")[1].strip())
print("="*30 + "\n")

# 3. Actualización del Repositorio Oficial
# Mantener el mismo nombre permite un historial de versiones (commits) limpio.
nombre_modelo_hub = "CEGTEdicion/Llama-3.2-1B-espanol"

print(f"--- Actualizando repositorio en Hugging Face: {nombre_modelo_hub} ---")

# Descomenta las líneas de abajo para realizar la subida oficial
# model.push_to_hub_merged(
#     nombre_modelo_hub, 
#     tokenizer, 
#     save_method = "merged_16bit", 
#     token = "TU_TOKEN_DE_HUGGINGFACE"
# )

print("--------------------------------------------------------")
print("Proceso finalizado. El repositorio ha sido actualizado.")
print("--------------------------------------------------------")
