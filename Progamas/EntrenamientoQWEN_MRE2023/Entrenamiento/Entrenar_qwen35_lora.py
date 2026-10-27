## Importaciones
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from Progamas.EntrenamientoQWEN_MRE2023.Entrenamiento.CargaCSV import cargar_csv
from Progamas.EntrenamientoQWEN_MRE2023.Entrenamiento.Contexto import enviar_contexto
from Progamas.EntrenamientoQWEN_MRE2023.Entrenamiento.MetalRiskDataset import MetalRiskDataset
from  Progamas.EntrenamientoQWEN_MRE2023.Entrenamiento.CausalCollator import  CausalCollator
from torch.utils.data import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer,
                          TrainingArguments,)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support,)



# ============================================================
# 1. CONFIGURACIÓN: AJUSTA AQUÍ LAS RUTAS
# ============================================================
# Cargar variables del archivo .env
load_dotenv()

## Diccionario de PATHs
PATHS = {
    # Tarea 2 
    "Terea2_train_2023": Path(os.environ["path_metalriskES2023_task2_CSV_train"]),
    "Terea2_trial_2023": Path(os.environ["path_metalriskES2023_task2_CSV_trial"]),
    "Tarea2_prediccion_2023" : Path(os.environ["path_predicciones_tarea2_A"]),
}

## Variables de rutas
# Tarea 2 rutas

TRAIN_CSV = PATHS["Terea2_train_2023"] / "Tarea_A.csv"
TRIAL_CSV = PATHS["Terea2_trial_2023"] / "Tarea_A.csv"

OUTPUT_DIR = PATHS("Tarea2_prediccion_2023")
ADAPTER_DIR = OUTPUT_DIR / "adaptador"
MODEL_ID = "Qwen/Qwen3.5-9B"
LABEL_COLUMN = "etiqueta_gold"
TEXT_COLUMN = "mensaje"
MAX_MESSAGE_TOKENS = 256
MAX_SEQ_LENGTH = 384
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
SEED = 42
SYSTEM_PROMPT  = pd.read_csv("Prompt.txt", sep="\t")


# ============================================================
# VALIDACIONES INICIALES
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible. Revisa la GPU.")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("La GPU no reporta soporte para BF16.")

for archivo in (TRAIN_CSV, TRIAL_CSV):
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {archivo}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GiB")
print("Modelo:", MODEL_ID)
print("Método: LoRA estándar, sin cuantización")
print("Precisión: BF16")


# ============================================================
# CARGA DE DATOS
# ============================================================

train_df = cargar_csv(TRAIN_CSV)
trial_df = cargar_csv(TRIAL_CSV)

print(f"\nMensajes Train: {len(train_df):,}")
print("Etiquetas Train:")
print(train_df[LABEL_COLUMN].value_counts().sort_index())

print(f"\nMensajes Trial: {len(trial_df):,}")
print("Etiquetas Trial:")
print(trial_df[LABEL_COLUMN].value_counts().sort_index())


# ============================================================
# TOKENIZADOR Y FORMATO DE CLASIFICACIÓN
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

END_IDS = tokenizer.encode(
    "<|im_end|>", add_special_tokens=False
)

if not END_IDS:
    raise RuntimeError("No se pudo identificar el token de fin de turno.")


print("\nTokenizando Train...")
train_dataset = MetalRiskDataset(train_df)

print("Tokenizando Trial...")
trial_dataset = MetalRiskDataset(trial_df)

print("Tokenización terminada.")


# ============================================================
# COLLATOR: RELLENO DE SECUENCIAS
# ============================================================
collator = CausalCollator(tokenizer.pad_token_id)


# ============================================================
# CARGA DEL MODELO Y CONFIGURACIÓN LoRA
# ============================================================

print("\nCargando Qwen3.5-9B en BF16...")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             torch_dtype=torch.bfloat16,
                                             device_map={"": 0},
                                             low_cpu_mem_usage=True,)

model.config.use_cache = False

lora_config = LoraConfig(task_type="CAUSAL_LM",
                         r=LORA_R,
                         lora_alpha=LORA_ALPHA,
                         lora_dropout=LORA_DROPOUT,
                         target_modules="all-linear",
                         bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Ayuda a reducir memoria durante el entrenamiento.
model.enable_input_require_grads()


# ============================================================
# ENTRENAMIENTO
# ============================================================

training_args = TrainingArguments(output_dir=str(OUTPUT_DIR / "checkpoints"),
                                  
                                  num_train_epochs=EPOCHS,
                                  per_device_train_batch_size=BATCH_SIZE,
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=GRADIENT_ACCUMULATION,                                    
                                  learning_rate=LEARNING_RATE,
                                  lr_scheduler_type="linear",
                                  warmup_ratio=0.03,
                                  weight_decay=0.01,
                                  optim="adamw_torch",
                                  
                                  bf16=True,
                                  fp16=False,
                                  
                                  gradient_checkpointing=True,
                                  gradient_checkpointing_kwargs={"use_reentrant": False},
                                  
                                  eval_strategy="epoch",
                                  save_strategy="epoch",
                                  logging_strategy="steps",
                                  logging_steps=10,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  greater_is_better=False,
                                  save_total_limit=2,
                                  
                                  report_to="none",
                                  seed=SEED,
                                  data_seed=SEED,
                                  remove_unused_columns=False,
                                  dataloader_num_workers=0,
                                )

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=trial_dataset,
                  data_collator=collator,
                  processing_class=tokenizer,)

print("\nIniciando entrenamiento...")
print("Train se usa para ajustar LoRA; Trial para validar.")

torch.cuda.reset_peak_memory_stats()

trainer.train()

print("\nEntrenamiento terminado.")
print("Mejor checkpoint:", trainer.state.best_model_checkpoint)

# Guarda el adaptador seleccionado por la validación.
trainer.save_model(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

print("Adaptador guardado en:", ADAPTER_DIR)


# ============================================================
# EVALUACIÓN POR MENSAJE EN TRIAL
# ============================================================

model = trainer.model
model.eval()
model.config.use_cache = True

device = next(model.parameters()).device


def predecir_mensaje(mensaje):
    message_ids = tokenizer.encode(mensaje,
                                   add_special_tokens=False,
                                   truncation=True,
                                   max_length=MAX_MESSAGE_TOKENS,
    )

    mensaje_recortado = tokenizer.decode(message_ids,
                                         skip_special_tokens=True,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Mensaje a clasificar:\n{mensaje_recortado}",
        },
    ]

    prompt_ids = tokenizer.apply_chat_template(messages,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=False,
                                               return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(prompt_ids)

    with torch.inference_mode():
        output = model.generate(input_ids=prompt_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=4,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=END_IDS[-1],
        )

    new_tokens = output[0, prompt_ids.shape[1]:]
    respuesta = tokenizer.decode(new_tokens,
                                 skip_special_tokens=True,
    ).strip()

    # La respuesta ideal es exclusivamente "0" o "1".
    if respuesta in {"0", "1"}:
        return int(respuesta), respuesta

    # Registramos respuestas inesperadas sin ocultarlas.
    match = re.match(r"^\s*([01])(?:\s|$)", respuesta)
    if match:
        return int(match.group(1)), respuesta

    return -1, respuesta


predicciones = []
salidas = []

print("\nEvaluando Trial...")
for i, mensaje in enumerate(trial_df[TEXT_COLUMN]):
    pred, salida = predecir_mensaje(mensaje)
    predicciones.append(pred)
    salidas.append(salida)

    if (i + 1) % 50 == 0 or (i + 1) == len(trial_df):
        print(f"Procesados: {i + 1}/{len(trial_df)}")


y_true = trial_df[LABEL_COLUMN].to_numpy()
y_pred = np.array(predicciones)

validos = np.isin(y_pred, [0, 1])
invalidos = int((~validos).sum())

print("\n" + "=" * 55)
print("RESULTADOS EN TRIAL")
print("=" * 55)
print("Total de mensajes:", len(y_true))
print("Predicciones válidas:", int(validos.sum()))
print("Respuestas inválidas:", invalidos)

if validos.any():
    y_true_valid = y_true[validos]
    y_pred_valid = y_pred[validos]

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_valid,
                                                               y_pred_valid,
                                                               labels=[0, 1],
                                                               average="macro",
                                                               zero_division=0,
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro-P:   {precision:.4f}")
    print(f"Macro-R:   {recall:.4f}")
    print(f"Macro-F1:  {f1:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_true_valid,
                                y_pred_valid,
                                labels=[0, 1],
                                target_names=["Control (0)", "Positivo (1)"],
                                digits=4,
                                zero_division=0,
    ))

    print("Matriz de confusión (filas=real, columnas=predicción):")
    print(confusion_matrix(y_true_valid,
                           y_pred_valid,
                           labels=[0, 1],))

# Guardamos las predicciones para poder analizarlas después.
resultados = pd.DataFrame({"fila_csv": np.arange(len(trial_df)),
                           "etiqueta_real": y_true,
                           "prediccion": y_pred,
                           "respuesta_modelo": salidas,
})

ruta_resultados = OUTPUT_DIR / "predicciones_trial.csv"
resultados.to_csv(ruta_resultados, index=False)

print("\nPredicciones guardadas en:", ruta_resultados)
print("Pico de VRAM:",
      round(torch.cuda.max_memory_allocated() / 1024**3, 2),
      "GiB")
print("\nExperimento finalizado.")
print("Test no fue utilizado.")
