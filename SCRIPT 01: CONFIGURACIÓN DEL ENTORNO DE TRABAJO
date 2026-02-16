# ==============================================================================
# SCRIPT 01: CONFIGURACIÓN DEL ENTORNO DE TRABAJO
# Propósito: Instalación de Unsloth, dependencias de CUDA y librerías de IA.
# Autor: CEGTEdicion
# ==============================================================================

# 1. Instalamos Unsloth: La herramienta principal de optimización.
# Utilizamos la versión compatible con CUDA 12.1 (estándar en Colab/Kaggle).
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Instalamos dependencias críticas para el manejo de tensores y optimización.
# - xformers: Acelera la atención en la GPU.
# - trl, peft, accelerate: El ecosistema de Hugging Face para Fine-Tuning.
# - bitsandbytes: Para la cuantización (reducir el peso del modelo).
pip install --no-deps xformers trl peft accelerate bitsandbytes

# 3. Instalamos librerías de visualización y métricas.
# Wandb nos servirá para monitorear el "Loss" en la nube.
pip install wandb matplotlib 

echo "--------------------------------------------------------"
echo "Entorno configurado con éxito. Listo para el entrenamiento."
echo "--------------------------------------------------------"
