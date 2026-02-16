# ==============================================================================
# SCRIPT 03: PROCESAMIENTO DE DATOS Y ENTRENAMIENTO
# Propósito: Formateo de instrucciones y ejecución del Fine-Tuning.
# ==============================================================================

import os
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# 1. Configuración del Formateador de Instrucciones (Alpaca Style)
# He ajustado el prompt para que sea más coherente con un tono académico.
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
        # Es vital que el EOS_TOKEN esté al final para que el modelo sepa cuándo callar
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 2. Preparación Inteligente del Dataset
# Asegúrate de que tu archivo se llame exactamente así en la carpeta de Colab
# 2. Preparación Inteligente del Dataset
data_file = "plncmm/spanish-alpaca" 

print(f"--- Descargando dataset desde Hugging Face: {data_file} ---")
# Cargamos directamente desde la nube
dataset = load_dataset(data_file, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# 3. Configuración de los Hiperparámetros
# He ajustado los pasos para una especialización más profunda (Saturación).
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
        warmup_steps = 10,       # Un poco más de calentamiento para estabilidad
        max_steps = 200,         # Aumentado ligeramente para mejor fijación de datos
        learning_rate = 5e-5,    
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", # Cambio a 'cosine' para una prosa más suave
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 4. Ejecución del Entrenamiento
trainer_stats = trainer.train()

print("--------------------------------------------------------")
print(f"Entrenamiento completado. Loss final: {trainer_stats.training_loss}")
print("--------------------------------------------------------")
