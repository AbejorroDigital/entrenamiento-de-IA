# ==============================================================================
# SCRIPT 03: PROCESAMIENTO DE DATOS Y ENTRENAMIENTO
# Propósito: Carga inteligente de datos y ejecución del bucle de entrenamiento.
# Autor: CEGTEdicion
# ==============================================================================

import os
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# 1. Configuración del Formateador de Instrucciones (Prompt Style)
alpaca_prompt = """Debajo hay una instrucción que describe una tarea. 
Escribe una respuesta que complete adecuadamente lo que se pide.

### Instrucción:
{}

### Respuesta:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 2. Preparación Inteligente del Dataset
# Reemplaza 'tu_dataset.json' con tu archivo (puede ser .json, .jsonl o .csv)
data_file = "tu_dataset.json" 

if os.path.exists(data_file):
    # Detectamos la extensión para elegir el motor de carga correcto
    extension = data_file.split(".")[-1]
    if extension == "jsonl": extension = "json" # load_dataset trata jsonl como json
    
    print(f"--- Cargando dataset desde {data_file} (Formato: {extension}) ---")
    dataset = load_dataset(extension, data_files=data_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
else:
    raise FileNotFoundError(f"No se encontró el archivo {data_file}. Asegúrate de subirlo a la ruta correcta.")

# 3. Configuración de los Hiperparámetros de Entrenamiento
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,             
        learning_rate = 2e-4,        
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 4. Ejecución del Entrenamiento
trainer_stats = trainer.train()

print("--------------------------------------------------------")
print(f"Entrenamiento completado. Loss final: {trainer_stats.training_loss}")
print("--------------------------------------------------------")
