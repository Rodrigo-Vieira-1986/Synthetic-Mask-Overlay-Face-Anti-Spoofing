"""
SCRIPT DE TREINAMENTO PARA DETECÇÃO DE SPOOFING FACIAL

DESCRIÇÃO GERAL:
Este script realiza o treinamento de modelos de deep learning para classificar imagens
faciais em duas categorias:
- bonafide: rostos reais
- attack: rostos falsos (ex: máscaras, spoofing)

O objetivo é criar modelos capazes de distinguir entre rostos reais e falsos,
para uso em sistemas de autenticação biométrica.

---------------------------------------------------------------------

ARQUITETURAS DISPONÍVEIS:
O usuário pode escolher qual modelo treinar:

1 - MobileNetV2
   - Mais leve
   - Mais rápido
   - Ideal para sistemas em tempo real

2 - ResNet50
   - Mais profundo
   - Melhor extração de features
   - Mais pesado

3 - Xception
   - Alta performance
   - Melhor separação de padrões complexos
   - Mais custoso computacionalmente

---------------------------------------------------------------------

ESTRUTURA DO DATASET (OBRIGATÓRIA):

Dataset/
 ├── train/
 │    ├── bonafide/
 │    └── attack/
 ├── val/
 │    ├── bonafide/
 │    └── attack/

Cada subpasta deve conter imagens correspondentes à classe.

---------------------------------------------------------------------

FLUXO DO TREINAMENTO:

1. O usuário escolhe o modelo a ser treinado
2. O dataset é carregado utilizando ImageDataGenerator
3. É aplicado data augmentation (variação de iluminação, zoom e rotação)
4. O modelo base é carregado com pesos pré-treinados (ImageNet)
5. As camadas base são congeladas (transfer learning)
6. O modelo é treinado monitorando val_loss
7. O melhor modelo é salvo automaticamente (checkpoint .h5)
8. Caso o treinamento seja interrompido:
   - os pesos do modelo são recuperados
   - o CSV é mantido
   - o treinamento continua da última época registrada

9. Após o treinamento inicial, é realizado fine-tuning:
   - últimas camadas são liberadas
   - o modelo se adapta melhor ao problema de spoofing

10. As métricas são salvas continuamente em CSV
11. O modelo final é salvo em formato .h5

---------------------------------------------------------------------

MÉTRICAS UTILIZADAS:

- accuracy  -> acurácia geral
- precision -> evita falsos positivos
- recall    -> detecta ataques corretamente
- AUC       -> qualidade geral da separação entre classes

---------------------------------------------------------------------

OBSERVAÇÕES IMPORTANTES:

- O treinamento usa EarlyStopping para parar automaticamente quando não há melhora
- O número de épocas (EPOCHS) é apenas um limite máximo
- O treino pode parar antes se o modelo convergir
- O checkpoint salva apenas o melhor modelo (menor val_loss)
- O CSVLogger salva métricas continuamente
- Em caso de interrupção (CTRL+C, queda de energia, etc),
  o treinamento continua da última época salva no CSV

---------------------------------------------------------------------

SAÍDAS GERADAS:

- mobilenet_fase1.csv
- mobilenet_finetuning.csv
- mobilenet.h5 (checkpoint)
- mobilenet_final.h5 (modelo final)

(O mesmo vale para resnet e xception)

---------------------------------------------------------------------
"""

import os
import json
import shutil
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50, Xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# SEÇÃO 1: CONFIGURAÇÃO
# =====================================================

# =====================================================
# ESCOLHA DO MODELO
# =====================================================
print("Escolha o modelo:")
print("1 - MobileNetV2")
print("2 - ResNet50")
print("3 - Xception")

opcao = input("Digite o número do modelo: ")

if opcao == "1":
    base_model_fn = MobileNetV2
    input_shape = (224, 224, 3)
    nome_modelo = "mobilenet"
    preprocess_fn = mobilenet_preprocess
elif opcao == "2":
    base_model_fn = ResNet50
    input_shape = (224, 224, 3)
    nome_modelo = "resnet"
    preprocess_fn = resnet_preprocess
elif opcao == "3":
    base_model_fn = Xception
    input_shape = (299, 299, 3)
    nome_modelo = "xception"
    preprocess_fn = xception_preprocess
else:
    raise ValueError("Opção inválida")

# =====================================================
# DEFINIÇÃO DOS CAMINHOS
# =====================================================

# script_dir SEMPRE deve existir (usado em checkpoints)
script_dir = os.path.dirname(os.path.abspath(__file__))

modelos_dir = os.path.join(script_dir, nome_modelo)
os.makedirs(modelos_dir, exist_ok=True)

# ============================
# OPÇÃO 1 - CAMINHO DINÂMICO (Mac / portátil)
# ============================
base_dir = os.path.join(script_dir, "Dataset")

# ============================
# OPÇÃO 2 - CAMINHO FIXO WINDOWS
# (descomente se estiver usando Windows)
# ============================
# Caminho da pasta Dataset contendo train/val/test
# base_dir = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Caminhos dos arquivos
ckpt_path = os.path.join(modelos_dir, f"{nome_modelo}.h5")

ckpt_finetuning_path = os.path.join(
    modelos_dir,
    f"{nome_modelo}_finetuning.h5"
)

csv_fase1_path = os.path.join(
    modelos_dir,
    f"{nome_modelo}_fase1.csv"
)

csv_finetuning_path = os.path.join(
    modelos_dir,
    f"{nome_modelo}_finetuning.csv"
)

ft_state_path = os.path.join(
    modelos_dir,
    f"{nome_modelo}_finetuning_state.json"
)

fase1_state_path = os.path.join(
    modelos_dir,
    f"{nome_modelo}_fase1_state.json"
)

# =====================================================
# CONFIGURAÇÕES DE TREINO
# =====================================================
BATCH_SIZE = 16
EPOCHS = 1000
FINE_TUNE_EPOCHS = 20

FINE_TUNE_LAYERS = {
    "mobilenet": 20,
    "resnet": 35,
    "xception": 45
}

# =====================================================
# SEÇÃO 2: DATASET
# =====================================================

# =====================================================
# DATA AUGMENTATION
# =====================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    brightness_range=[0.6, 1.4],
    zoom_range=0.1,
    rotation_range=10
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# =====================================================
# BALANCEAMENTO DAS CLASSES
# =====================================================

class_counts = {}
class_indices = train_gen.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

for cls_name, cls_idx in class_indices.items():
    class_counts[cls_name] = np.sum(train_gen.classes == cls_idx)

print("\nDistribuição das classes (treino):")
for cls, cnt in class_counts.items():
    print(f"  {cls}: {cnt} imagens")

values = list(class_counts.values())
if len(values) >= 2:
    ratio = max(values) / min(values)
    usar_class_weight = ratio > 1.5
else:
    usar_class_weight = False

if usar_class_weight:
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(zip(np.unique(train_gen.classes), weights))

    print("\nPesos calculados para cada classe:")
    for idx, w in sorted(class_weight_dict.items()):
        cls_name = index_to_class[idx]
        print(f"  {cls_name} (índice {idx}): {w:.4f}")
    print("\n-> class_weight será utilizado no treinamento")
else:
    class_weight_dict = None
    print("\nClasses balanceadas. class_weight não será utilizado.")
print()

# =====================================================
# SEÇÃO 3: CONSTRUÇÃO DO MODELO
# =====================================================
base_model = base_model_fn(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)

# Transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Cabeça da rede
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# =====================================================
# CARREGAR CHECKPOINT
# =====================================================

# =====================================================
# FUNÇÕES AUXILIARES - ESTADO DO TREINAMENTO
# =====================================================

def salvar_estado_ft(path, epoch, best_loss, wait, best_epoch=0):
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "wait": wait,
        "best_epoch": best_epoch
    }
    with open(path, "w") as f:
        json.dump(state, f)


def carregar_estado_ft(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _obter_epoch_backup(backup_dir):
    ckpt_path = os.path.join(backup_dir, "checkpoint")
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, "r") as f:
        for line in f:
            if "model_checkpoint_path" in line:
                for part in line.split('"'):
                    if part.startswith("ckpt-"):
                        return int(part.split("-")[1])
    return None


class FineTuningState(tf.keras.callbacks.Callback):
    def __init__(self, state_path, early_stop):
        super().__init__()
        self.state_path = state_path
        self.early_stop = early_stop
        self._best_loss = None
        self._best_epoch = None

    def on_train_begin(self, logs=None):
        estado = carregar_estado_ft(self.state_path)
        if estado is not None:
            preserved = estado.get("best_loss")
            if preserved is not None and np.isfinite(float(preserved)):
                self._best_loss = float(preserved)
            preserved_epoch = estado.get("best_epoch")
            if preserved_epoch is not None:
                self._best_epoch = int(preserved_epoch)

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.early_stop.monitor)
        if current is not None and np.isfinite(float(current)):
            current = float(current)
            if self._best_loss is None or self.early_stop.monitor_op(current, self._best_loss):
                self._best_loss = current
                self._best_epoch = epoch + 1
        if self._best_loss is not None:
            salvar_estado_ft(
                self.state_path,
                epoch + 1,
                self._best_loss,
                self.early_stop.wait,
                self._best_epoch
            )


# =====================================================
# SEÇÃO 4: FASE 1
# =====================================================

# =====================================================
# COMPILAÇÃO
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# =====================================================
# CALLBACKS - FASE 1
# =====================================================
csv_logger_fase1 = tf.keras.callbacks.CSVLogger(
    csv_fase1_path,
    append=True
)

backup_fase1 = tf.keras.callbacks.BackupAndRestore(
    backup_dir=os.path.join(modelos_dir, f"{nome_modelo}_backup_fase1"),
    save_freq='epoch',
    delete_checkpoint=True
)

# =====================================================
# CONTINUAR DA ÚLTIMA ÉPOCA (FASE 1)
# =====================================================

fase1_state = carregar_estado_ft(fase1_state_path)
backup_dir_fase1 = os.path.join(modelos_dir, f"{nome_modelo}_backup_fase1")
epoch_backup = _obter_epoch_backup(backup_dir_fase1)

if fase1_state is not None:
    epoch_state = int(fase1_state["epoch"])
    if epoch_backup is not None:
        if epoch_backup == epoch_state - 1:
            print("Fase 1: estados consistente (BackupAndRestore + state.json).")
            initial_epoch = epoch_state
            initial_best = float(fase1_state["best_loss"]) if fase1_state["best_loss"] is not None else None
            initial_wait = int(fase1_state["wait"])
            initial_best_epoch = int(fase1_state.get("best_epoch", 0))
        else:
            epoch_final_0 = min(epoch_backup, epoch_state - 1)
            initial_epoch = epoch_final_0 + 1
            initial_best = None
            initial_wait = 0
            initial_best_epoch = 0
            salvar_estado_ft(fase1_state_path, initial_epoch, None, 0, 0)
            print(f"Aviso: inconsistência BackupAndRestore (época {epoch_backup}) vs "
                  f"state.json (época {epoch_state}).")
            print(f"Usando época {initial_epoch} como ponto mais recente consistente.")
    else:
        initial_epoch = epoch_state
        initial_best = float(fase1_state["best_loss"]) if fase1_state["best_loss"] is not None else None
        initial_wait = int(fase1_state["wait"])
        initial_best_epoch = int(fase1_state.get("best_epoch", 0))
        print(f"Fase 1: retomando de estado salvo (época {initial_epoch}).")
else:
    initial_best = None
    initial_wait = 0
    initial_epoch = 0
    initial_best_epoch = 0

# =====================================================
# VERIFICAR SE FASE 1 JÁ FOI CONCLUÍDA
# =====================================================
PACiencia = 10
fase1_concluida = False
if fase1_state is not None and initial_wait >= PACiencia:
    fase1_concluida = True
    print("Fase 1 já foi concluída (EarlyStopping atingiu patience).")
    print(f"wait ({initial_wait}) >= patience ({PACiencia})")
    if os.path.exists(ckpt_path):
        print(f"Carregando melhor modelo da Fase 1: {ckpt_path}")
        model.load_weights(ckpt_path)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    initial_value_threshold=initial_best
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

if initial_best is not None:
    early_stop.best = initial_best
    early_stop.wait = initial_wait

fase1_state_callback = FineTuningState(
    state_path=fase1_state_path,
    early_stop=early_stop
)

# =====================================================
# TREINAMENTO FASE 1
# =====================================================
inicio = time.time()

if not fase1_concluida:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[
            backup_fase1,
            checkpoint,
            early_stop,
            csv_logger_fase1,
            fase1_state_callback
        ],
        verbose=1,
        class_weight=class_weight_dict
    )
else:
    print("Fase 1 já concluída. Pulando treinamento da Fase 1.")

# =====================================================
# SEÇÃO 5: FINE-TUNING
# =====================================================
print("\nIniciando fine-tuning...\n")

# Garante que o Fine-Tuning sempre inicie do melhor modelo da Fase 1
if os.path.exists(ckpt_path):
    print(f"Carregando melhores pesos da Fase 1: {ckpt_path}")
    model.load_weights(ckpt_path)

# Libera últimas camadas (específico por arquitetura)
n_layers_ft = FINE_TUNE_LAYERS.get(nome_modelo, 20)
for layer in base_model.layers[-n_layers_ft:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)



# =====================================================
# CALLBACKS - FINE-TUNING
# =====================================================

csv_logger_finetuning = tf.keras.callbacks.CSVLogger(
    csv_finetuning_path,
    append=True
)

backup_finetuning = tf.keras.callbacks.BackupAndRestore(
    backup_dir=os.path.join(modelos_dir, f"{nome_modelo}_backup_finetuning"),
    save_freq='epoch',
    delete_checkpoint=True
)

# =====================================================
# CONTINUAR FINE-TUNING
# =====================================================

ft_state = carregar_estado_ft(ft_state_path)
backup_dir_ft = os.path.join(modelos_dir, f"{nome_modelo}_backup_finetuning")

# Remove backup obsoleto se não há arquivo de estado,
# evitando que BackupAndRestore sobrescreva o initial_epoch_ft
if ft_state is None and os.path.exists(backup_dir_ft):
    shutil.rmtree(backup_dir_ft)
    print(f"Backup obsoleto do Fine-Tuning removido: {backup_dir_ft}")

epoch_backup = _obter_epoch_backup(backup_dir_ft)

if ft_state is not None:
    epoch_state = int(ft_state["epoch"])
    if epoch_backup is not None:
        if epoch_backup == epoch_state - 1:
            print("Fine-Tuning: estados consistente (BackupAndRestore + state.json).")
            initial_epoch_ft = epoch_state
            initial_best = float(ft_state["best_loss"]) if ft_state["best_loss"] is not None else None
            initial_wait = int(ft_state["wait"])
            initial_best_epoch = int(ft_state.get("best_epoch", 0))
        else:
            epoch_final_0 = min(epoch_backup, epoch_state - 1)
            initial_epoch_ft = epoch_final_0 + 1
            initial_best = None
            initial_wait = 0
            initial_best_epoch = 0
            salvar_estado_ft(ft_state_path, initial_epoch_ft, None, 0, 0)
            print(f"Aviso: inconsistência BackupAndRestore (época {epoch_backup}) vs "
                  f"state.json (época {epoch_state}).")
            print(f"Usando época {initial_epoch_ft} como ponto mais recente consistente.")
    else:
        initial_epoch_ft = epoch_state
        initial_best = float(ft_state["best_loss"]) if ft_state["best_loss"] is not None else None
        initial_wait = int(ft_state["wait"])
        initial_best_epoch = int(ft_state.get("best_epoch", 0))
        print(f"Fine-Tuning: retomando de estado salvo (época {initial_epoch_ft}).")
else:
    initial_best = None
    initial_wait = 0
    initial_epoch_ft = 0
    initial_best_epoch = 0

checkpoint_finetuning = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_finetuning_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    initial_value_threshold=initial_best
)

early_stop_finetuning = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

if initial_best is not None:
    early_stop_finetuning.best = initial_best
    early_stop_finetuning.wait = initial_wait

ft_state_callback = FineTuningState(
    state_path=ft_state_path,
    early_stop=early_stop_finetuning
)

# =====================================================
# TREINAMENTO FINE-TUNING
# =====================================================
if initial_epoch_ft < FINE_TUNE_EPOCHS:
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        initial_epoch=initial_epoch_ft,
        callbacks=[
            backup_finetuning,
            checkpoint_finetuning,
            early_stop_finetuning,
            csv_logger_finetuning,
            ft_state_callback
        ],
        verbose=1,
        class_weight=class_weight_dict
    )
else:
    print("Fine-Tuning já concluído anteriormente.")

fim = time.time()

print(
    f"Tempo total de treino: "
    f"{(fim - inicio)/60:.2f} minutos"
)

# =====================================================
# SEÇÃO 6: SALVAMENTO
# =====================================================

# =====================================================
# SALVAR MODELO FINAL
# =====================================================
if os.path.exists(ckpt_finetuning_path):
    print(f"Carregando melhores pesos do fine-tuning: {ckpt_finetuning_path}")
    model.load_weights(ckpt_finetuning_path)

model.save(os.path.join(modelos_dir, f"{nome_modelo}_final.h5"))

# =====================================================
# LIMPEZA DE ARQUIVOS TEMPORÁRIOS DO FINE-TUNING
# =====================================================
if os.path.exists(ft_state_path):
    os.remove(ft_state_path)
    print(f"  Removido: {ft_state_path}")

# =====================================================
# SEÇÃO 7: FINALIZAÇÃO
# =====================================================

print("Treinamento concluído")
