# ==============================================================================
# SCRIPT 01: CONFIGURACIÓN DEL ENTORNO DE TRABAJO
# Propósito: Instalación de Unsloth y ecosistema de IA.
# Nota: Se han añadido los prefijos '!' para ejecución en shell de Colab.
# ==============================================================================

# 1. Instalamos Unsloth con optimización para CUDA.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Instalamos dependencias críticas de Hugging Face y aceleración.
!pip install --no-deps xformers trl peft accelerate bitsandbytes

# 3. Instalamos librerías de monitoreo y graficación.
!pip install wandb matplotlib 

# 4. Verificación visual del éxito de la operación.
!echo "--------------------------------------------------------"
!echo "Entorno configurado con éxito. Listo para el entrenamiento."
!echo "--------------------------------------------------------"
