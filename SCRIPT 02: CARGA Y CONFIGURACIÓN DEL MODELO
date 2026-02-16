# ==============================================================================
# SCRIPT 02: CARGA Y CONFIGURACIÓN DEL MODELO
# Propósito: Descargar Llama 3.2-1B y aplicar cuantización de 4 bits.
# Autor: CEGTEdicion
# ==============================================================================

from unsloth import FastLanguageModel
import torch

# 1. Definición de variables de configuración
model_name = "unsloth/Llama-3.2-1B-instruct-bnb-4bit" # Versión optimizada
max_seq_length = 2048                              # Capacidad de contexto
dtype = None                                       # Detección automática de hardware
load_in_4bit = True                                # Activamos la compresión

# 2. Carga del modelo y el tokenizador
# Este proceso descarga los pesos desde Hugging Face y los prepara en la VRAM.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Aplicación de PEFT (Parameter-Efficient Fine-Tuning) mediante LoRA
# En lugar de entrenar todos los parámetros, solo añadimos capas adaptables.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,            # Rango de las matrices de adaptación (Rank)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,   # Factor de escala
    lora_dropout = 0,  # Optimizado para 0 para mayor velocidad
    bias = "none",     # No entrenamos los sesgos para ahorrar memoria
    use_gradient_checkpointing = "unsloth", # Reduce uso de memoria en un 70%
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print("--------------------------------------------------------")
print(f"Modelo {model_name} cargado y configurado con LoRA.")
print("Listo para recibir el dataset.")
print("--------------------------------------------------------")
