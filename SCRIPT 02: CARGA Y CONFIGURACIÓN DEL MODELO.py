# ==============================================================================
# SCRIPT 02: CARGA Y CONFIGURACIÓN DEL MODELO
# Propósito: Descargar Llama 3.2-1B y configurar la arquitectura LoRA.
# ==============================================================================

from unsloth import FastLanguageModel
import torch

# 1. Definición de variables de configuración
model_name = "unsloth/Llama-3.2-1B-instruct-bnb-4bit"
max_seq_length = 2048   # Puedes subirlo a 4096 si tus ejemplos son muy largos
dtype = None            # Autodetección (float16 para T4, bfloat16 para A100)
load_in_4bit = True     # Cuantización de 4 bits activada

# 2. Carga del modelo y el tokenizador
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Aplicación de LoRA para Especialización Profunda
# Nota: Incluimos todos los módulos para forzar el aprendizaje factual.
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,            # Subí el rango a 32 para capturar mejor tu prosa académica
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,   # Alpha igual al rango suele dar estabilidad
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print("--------------------------------------------------------")
print(f"Modelo {model_name} cargado y configurado con LoRA.")
print("Listo para recibir el dataset.")
print("--------------------------------------------------------")
