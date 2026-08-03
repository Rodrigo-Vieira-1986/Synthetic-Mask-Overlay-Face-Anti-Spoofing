"""
ETAPA_6_INFERENCIA_TRANSFORMERS_V2.PY
================================================================================
Script de inferência para modelos Transformer de Face Anti-Spoofing.

Descrição:
    Este script realiza a avaliação experimental completa de modelos
    Transformer (ViT, DeiT, Swin) treinados para detecção de spoofing
    facial (ataque vs. bonafide). Carrega os modelos salvos (.pth),
    executa inferência no conjunto de teste, calcula métricas de
    classificação e gera relatórios e gráficos.

Arquiteturas suportadas:
    - ViT-B16     (vit_base_patch16_224, entrada: 224x224)
    - DeiT-Small  (deit_small_patch16_224, entrada: 224x224)
    - Swin-Tiny   (swin_tiny_patch4_window7_224, entrada: 224x224)

Métricas calculadas:
    Accuracy, Precision, Recall, F1-Score, ROC AUC, Specificity,
    HTER (Half Total Error Rate), EER (Equal Error Rate).

Arquivos gerados:
    - comparativo_test.csv            (métricas agregadas por modelo)
    - resumo_final.csv                (resumo ordenado por desempenho)
    - predicoes_{modelo}.csv          (predições individuais)
    - erros_{modelo}.csv              (apenas erros de classificação)
    - matriz_{modelo}.png             (matriz de confusão)
    - roc_{modelo}.png                (curva ROC)
    - classification_report_{modelo}.txt

Alterações da V2 (em relação à V1 original):
    1. Desempacotamento da matriz de confusão corrigido de
       tn, fp, fn, tp para tp, fn, fp, tn, garantindo que
       Specificity, FAR e FRR sejam calculados corretamente.

Metodologia:
    As métricas seguem exatamente as mesmas definições do treinamento
    (Etapa_5_Treinamento_Transformers_v2.py). O pré-processamento de
    teste replica as transformações de validação: Resize(224) ->
    ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]).

Contexto acadêmico:
    Script desenvolvido para dissertação de mestrado em Face Anti-Spoofing.

Autor:
    Rodrigo Santos

Versão:
    V2 — Documentada
================================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================
import os
import sys
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# =====================================================================
# SECAO 1: CONFIGURACAO (CAMINHOS, REPRODUTIBILIDADE, DISPOSITIVO)
# =====================================================================

# Suprime avisos não essenciais para manter a saída limpa
warnings.filterwarnings("ignore")

# --- Caminhos dos diretorios ----------------------------------------

# ================== CONFIGURE AQUI ==================
# Caminho da pasta com os modelos Transformers treinados (.pth)
MODELS_DIR = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_MODELOS_PTH"
# Caminho da pasta Dataset/test com as imagens de teste
DATASET_DIR = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET_TEST"
# Caminho da pasta onde serao salvos os resultados da inferencia Transformers
OUTPUT_DIR = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_RESULTADOS_TRANSFORMERS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Semente aleatoria para reproducibilidade -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Deteccao automatica do dispositivo disponivel -------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Dispositivo: MPS (Mac)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Dispositivo: CUDA ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("Dispositivo: CPU")

# --- Hiperparametros e constantes ------------------------------------
IMG_SIZE = 224
if DEVICE.type == "cuda":
    BATCH_SIZE = 64
elif DEVICE.type == "mps":
    BATCH_SIZE = 16
else:
    BATCH_SIZE = 8
NUM_CLASSES = 2
CLASS_NAMES = ["attack", "bonafide"]
CLASS_TO_IDX = {"attack": 0, "bonafide": 1}
POSITIVE_CLASS = 0  # attack e a classe positiva (deteccao de spoofing)

# Mapeamento: nome do modelo no checkpoint -> nome de exibicao
ARQUITETURAS = {
    "vit_base_patch16_224": "ViT-B16",
    "deit_small_patch16_224": "DeiT-Small",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
}

# Mapeamento: prefixo do nome do arquivo para nome de exibicao
PREFIXO_ARQUIVO = {
    "vit": "ViT-B16",
    "deit": "DeiT-Small",
    "swin": "Swin-Tiny",
}

ARQUITETURA_TIPO = "Transformer"

# =====================================================================
# SECAO 2: PRE-PROCESSAMENTO (TRANSFORMACOES DE TESTE)
# =====================================================================
# As transformacoes replicam exatamente o pipeline de validacao usado
# durante o treinamento (Etapa_5_Treinamento_Transformers_v2.py):
#   1. Redimensionamento para 224x224 (Resize)
#   2. Conversao para tensor (ToTensor: escala [0,255] -> [0,1])
#   3. Normalizacao com medias e desvios padrao do ImageNet
# =====================================================================

normalizacao_imagenet = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalizacao_imagenet,
])

# =====================================================================
# SECAO 3: FUNCOES AUXILIARES (FORMATACAO, PROGRESSO, MODELOS)
# =====================================================================


def _fmt_num(n):
    """Formata números grandes com separador de milhares no padrão brasileiro.

    Args:
        n (int or float): Número a ser formatado.

    Returns:
        str: Número formatado com ponto como separador de milhares.
    """
    return f"{n:,}".replace(",", ".")


def _criar_barra_progresso(pct, largura=30):
    """Cria uma barra de progresso visual com caracteres Unicode.

    Args:
        pct (float): Porcentagem concluída (0 a 100).
        largura (int): Largura da barra em caracteres.

    Returns:
        str: Barra de progresso formatada.
    """
    preenchido = int(pct / 100.0 * largura)
    vazio = largura - preenchido
    return "[" + "\u2588" * preenchido + "\u2591" * vazio + "]"


def _formatar_linha_progresso(nome, processadas, total, tempo_decorrido, tempo_por_img):
    """Formata uma linha de progresso com barra, porcentagem e estimativas de tempo.

    Args:
        nome (str): Nome do modelo.
        processadas (int): Quantidade de imagens processadas.
        total (int): Total de imagens a processar.
        tempo_decorrido (float): Tempo decorrido em segundos.
        tempo_por_img (float): Tempo médio por imagem em segundos.

    Returns:
        str: Linha de progresso formatada para terminal.
    """
    pct = (processadas / total) * 100.0
    eta = (total - processadas) * tempo_por_img
    barra = _criar_barra_progresso(pct)
    h = int(tempo_decorrido // 3600)
    m = int((tempo_decorrido % 3600) // 60)
    s = int(tempo_decorrido % 60)
    eh = int(eta // 3600)
    em = int((eta % 3600) // 60)
    es = int(eta % 60)
    return (f"{nome} | {barra} {pct:5.1f}% | "
            f"{_fmt_num(processadas):>8}/{_fmt_num(total)} | "
            f"Tempo {h:02d}:{m:02d}:{s:02d} | "
            f"ETA {eh:02d}:{em:02d}:{es:02d} | "
            f"{tempo_por_img:.3f}s/img")


def identificar_arquitetura(checkpoint):
    """Identifica a arquitetura do modelo a partir do checkpoint salvo.

    A identificação é feita por dois métodos em cascata:
      1. Busca o campo 'model_name' no dicionário ARQUITETURAS.
      2. Fallback: verifica prefixos conhecidos no nome do modelo.

    Args:
        checkpoint (dict): Dicionário do checkpoint carregado via torch.load.
                           Deve conter a chave 'model_name' com o nome
                           da arquitetura timm utilizada no treinamento.

    Returns:
        tuple: (arquitetura_nome, model_name)
            - arquitetura_nome (str): Nome de exibição (ex.: "ViT-B16").
            - model_name (str): Nome técnico da arquitetura timm.
    """
    model_name = checkpoint.get("model_name", "")
    if model_name in ARQUITETURAS:
        return ARQUITETURAS[model_name], model_name

    for prefixo, nome in PREFIXO_ARQUIVO.items():
        if prefixo in str(model_name).lower():
            return nome, model_name

    # Fallback: tenta inferir pelo nome do arquivo
    for prefixo, nome in PREFIXO_ARQUIVO.items():
        if prefixo in str(model_name).lower():
            return nome, model_name

    return "Desconhecido", model_name


def contar_parametros(model):
    """Conta os parâmetros totais e treináveis do modelo PyTorch.

    Args:
        model (torch.nn.Module): Modelo carregado.

    Returns:
        tuple: (total_params, trainable_params)
            - total_params (int): Número total de parâmetros.
            - trainable_params (int): Número de parâmetros treináveis.
    """
    total = sum(p.numel() for p in model.parameters())
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, treinaveis


def carregar_modelo(caminho_pth):
    """Carrega um modelo .pth e retorna o modelo, arquitetura e metadados.

    O checkpoint deve ter sido salvo pelo script de treinamento
    (Etapa_5_Treinamento_Transformers_v2.py) e conter as chaves:
    'model_name', 'model_state_dict', 'class_to_idx', 'num_classes'.

    Args:
        caminho_pth (str): Caminho absoluto para o arquivo .pth.

    Returns:
        tuple: (model, arquitetura_nome, info)
            - model (torch.nn.Module): Modelo carregado e em modo eval.
            - arquitetura_nome (str): Nome de exibição da arquitetura.
            - info (dict): Metadados do checkpoint (class_names, positive_idx, etc.).

    Raises:
        FileNotFoundError: Se o arquivo .pth não existir.
        KeyError: Se o checkpoint não contiver 'model_state_dict'.
    """
    if not os.path.isfile(caminho_pth):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_pth}")

    checkpoint = torch.load(caminho_pth, map_location="cpu", weights_only=False)
    arquitetura_nome, model_name = identificar_arquitetura(checkpoint)

    if model_name not in ARQUITETURAS:
        # Tenta carregar mesmo que a arquitetura nao esteja no mapeamento
        print(f"  Aviso: arquitetura '{model_name}' nao reconhecida. Tentando carregar...")

    # Obtem num_classes do checkpoint (compativel com o treinamento)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    # Cria o modelo com timm
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )

    # Carrega os pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Reconstroi nomes das classes a partir do checkpoint
    class_to_idx = checkpoint.get("class_to_idx", CLASS_TO_IDX)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    info = {
        "arquitetura": arquitetura_nome,
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "class_names": class_names,
        "positive_idx": checkpoint.get("positive_idx", POSITIVE_CLASS),
        "num_classes": num_classes,
        "input_size": checkpoint.get("input_size", IMG_SIZE),
    }

    return model, arquitetura_nome, info


def criar_dataloader_teste():
    """Cria o DataLoader para o conjunto de teste com configuração otimizada.

    O dataset de teste deve seguir a estrutura:
        test/
        +-- attack/
        +-- bonafide/

    A configuração do DataLoader (batch_size, num_workers, pin_memory)
    é ajustada automaticamente conforme o dispositivo disponível.

    Returns:
        tuple: (loader, dataset)
            - loader (DataLoader): Iterável sobre os batches de teste.
            - dataset (ImageFolder): Dataset com acesso direto às amostras.

    Raises:
        FileNotFoundError: Se o diretório de teste não existir ou não
                           tiver a estrutura esperada.
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Diretorio de teste nao encontrado: {DATASET_DIR}\n"
            f"Estrutura esperada:\n"
            f"  {DATASET_DIR}/\n"
            f"  +-- attack/\n"
            f"  +-- bonafide/"
        )

    if DEVICE.type == "cuda":
        num_workers = max(2, os.cpu_count() // 2)
    else:
        num_workers = 2
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda"
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    dataset = datasets.ImageFolder(DATASET_DIR, transform_test)
    print(f"  Classes encontradas: {dataset.class_to_idx}")
    print(f"  Total de imagens: {len(dataset)}")
    print(f"  Workers: {num_workers}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    return loader, dataset


@torch.no_grad()
def inferir(model, loader, modelo_nome="", inicio_inferencia=None):
    """Executa inferência em lotes (batches) sobre todas as imagens do loader.

    O decorador @torch.no_grad() desabilita o cálculo de gradientes,
    reduzindo o consumo de memória e acelerando a inferência.

    Args:
        model (torch.nn.Module): Modelo em modo eval.
        loader (DataLoader): DataLoader com as imagens de teste.
        modelo_nome (str): Nome do modelo para exibição na barra de progresso.
        inicio_inferencia (float, optional): Timestamp de início (time.time())
            para cálculo de tempo total. Se None, usa o momento da chamada.

    Returns:
        tuple: (y_true, y_pred, y_prob, tempos_ms)
            - y_true (np.ndarray): Rótulos verdadeiros (N,).
            - y_pred (np.ndarray): Rótulos preditos (N,).
            - y_prob (np.ndarray): Probabilidades por classe (N, 2).
            - tempos_ms (np.ndarray): Tempo de inferência por imagem (N,) em ms.
    """
    model.eval()

    if inicio_inferencia is None:
        inicio_inferencia = time.time()

    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    tempos = []

    total_imagens = len(loader.dataset)
    processadas = 0

    for imagens, labels in loader:
        imagens = imagens.to(DEVICE)
        labels = labels.to(DEVICE)

        # Mede tempo da inferencia no batch
        inicio = time.perf_counter()
        saida = model(imagens)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        fim = time.perf_counter()

        tempo_batch = (fim - inicio) * 1000.0  # ms
        tempo_por_imagem = tempo_batch / imagens.size(0)
        tempos.extend([tempo_por_imagem] * imagens.size(0))

        probs = torch.softmax(saida, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true_list.extend(labels.cpu().numpy())
        y_pred_list.extend(preds.cpu().numpy())
        y_prob_list.extend(probs.cpu().numpy())

        processadas += imagens.size(0)
        tempo_decorrido = time.time() - inicio_inferencia
        tempo_medio = tempo_decorrido / processadas
        linha = _formatar_linha_progresso(modelo_nome, processadas, total_imagens, tempo_decorrido, tempo_medio)
        print(linha.ljust(130), end="\r", flush=True)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)
    tempos = np.array(tempos)

    return y_true, y_pred, y_prob, tempos


def calcular_metricas(y_true, y_pred, y_prob, positive_idx=0):
    """Calcula as métricas principais de classificação binária.

    A classe 'attack' (índice 0) é considerada a classe positiva, pois o
    objetivo do sistema de Face Anti-Spoofing é detectar corretamente os
    ataques (spoofing). As métricas de Precision, Recall e F1 são calculadas
    sobre a classe positiva.

    Args:
        y_true (np.ndarray): Rótulos verdadeiros (0=attack, 1=bonafide).
        y_pred (np.ndarray): Rótulos preditos (0=attack, 1=bonafide).
        y_prob (np.ndarray): Matriz de probabilidades (N, 2) onde
                             y_prob[:,0] = prob. de attack,
                             y_prob[:,1] = prob. de bonafide.
        positive_idx (int): Índice da classe positiva (default: 0 = attack).

    Returns:
        tuple: (acc, prec, rec, f1, auc, y_true_bin, y_pred_bin)
            - acc (float): Acurácia.
            - prec (float): Precision para a classe positiva.
            - rec (float): Recall para a classe positiva.
            - f1 (float): F1-Score para a classe positiva.
            - auc (float): ROC AUC.
            - y_true_bin (np.ndarray): Rótulos binarizados (1 = positiva).
            - y_pred_bin (np.ndarray): Predições binarizadas (1 = positiva).
    """
    # Probabilidade da classe positiva (usada para ROC AUC).
    # Esta coluna e extraida da matriz y_prob para calcular a curva ROC.
    y_prob_pos = y_prob[:, positive_idx]

    # Binarizacao: 1 se for a classe positiva (attack), 0 caso contrario
    y_true_bin = np.where(y_true == positive_idx, 1, 0)
    y_pred_bin = np.where(y_pred == positive_idx, 1, 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        auc = roc_auc_score(y_true_bin, y_prob_pos)
    except ValueError:
        auc = 0.0

    return acc, prec, rec, f1, auc, y_true_bin, y_pred_bin


def calcular_metricas_biometricas(cm, y_true_bin, y_prob_pos):
    """Calcula métricas biométricas a partir da matriz de confusão.

    Métricas calculadas:
      - Specificity (TNR): TN / (TN + FP)
        Proporção de bonafides corretamente identificados.
      - FAR (False Acceptance Rate): FP / (FP + TN)
        Proporção de bonafides incorretamente aceitos como ataque.
      - FRR (False Rejection Rate): FN / (FN + TP)
        Proporção de ataques incorretamente rejeitados como bonafide.
      - HTER (Half Total Error Rate): (FAR + FRR) / 2
        Média entre FAR e FRR, métrica padrão em sistemas biométricos.
      - EER (Equal Error Rate): Ponto onde FAR = FRR
        Medida única de desempenho do sistema biométrico.

    Notes:
        A matriz de confusão gerada pelo sklearn segue a convenção:
        confusion_matrix(labels=[0, 1]) retorna [[TP, FN], [FP, TN]]
        quando 0 = classe positiva (attack) e 1 = classe negativa (bonafide).
        Portanto, o desempacotamento correto é: tp, fn, fp, tn = cm.ravel().
        Este foi o principal erro corrigido na V2, que anteriormente usava
        tn, fp, fn, tp, resultando em métricas incorretas.

    Args:
        cm (np.ndarray): Matriz de confusão 2x2.
        y_true_bin (np.ndarray): Rótulos binarizados (1 = positiva).
        y_prob_pos (np.ndarray): Probabilidades da classe positiva.

    Returns:
        tuple: (specificity, hter, eer)
            - specificity (float): Specificity (True Negative Rate).
            - hter (float): Half Total Error Rate.
            - eer (float): Equal Error Rate.
    """
    # V2: desempacotamento corrigido da matriz de confusao.
    # sklearn confusion_matrix(labels=[0, 1]) retorna [[TP, FN], [FP, TN]].
    # Portanto cm.ravel() = [tp, fn, fp, tn] para labels [0=attack, 1=bonafide].
    tp, fn, fp, tn = cm.ravel()

    denom_spec = tn + fp
    specificity = tn / denom_spec if denom_spec > 0 else 0.0

    denom_far = fp + tn
    far = fp / denom_far if denom_far > 0 else 0.0
    denom_frr = fn + tp
    frr = fn / denom_frr if denom_frr > 0 else 0.0
    hter = (far + frr) / 2.0

    try:
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_pos)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
    except Exception:
        eer = 0.0

    return specificity, hter, eer


def calcular_tempo(tempos_ms):
    """Calcula estatísticas descritivas dos tempos de inferência.

    Args:
        tempos_ms (np.ndarray): Array com os tempos de inferência em
                                milissegundos para cada imagem.

    Returns:
        dict: Dicionário com média, desvio, mínimo, máximo e total dos tempos.
    """
    return {
        "media": float(np.mean(tempos_ms)),
        "desvio": float(np.std(tempos_ms)),
        "min": float(np.min(tempos_ms)),
        "max": float(np.max(tempos_ms)),
        "total": float(np.sum(tempos_ms)),
    }


def gerar_matriz_confusao(cm, classes, titulo, caminho_salvar):
    """Gera e salva a matriz de confusão como imagem PNG.

    A matriz de confusão é gerada com o sklearn, que retorna a matriz na
    ordem [[TP, FN], [FP, TN]] para labels [0=attack, 1=bonafide].
    A visualização apresenta as classes nos eixos X (predito) e Y (real).
    Os valores numéricos são exibidos no centro de cada célula, com cor
    do texto ajustada automaticamente para contraste.

    Args:
        cm (np.ndarray): Matriz de confusão 2x2.
        classes (list): Nomes das classes para os rótulos dos eixos.
        titulo (str): Título do gráfico.
        caminho_salvar (str): Caminho para salvar a imagem PNG.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(titulo, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.ylabel("Rotulo Verdadeiro", fontsize=12)
    plt.xlabel("Rotulo Predito", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz de confusao salva: {caminho_salvar}")


def gerar_curva_roc(y_true_bin, y_prob_pos, nome_modelo, caminho_salvar):
    """Gera e salva a curva ROC como imagem PNG.

    A curva ROC (Receiver Operating Characteristic) plota a FPR
    (1 - Specificity) no eixo X versus a TPR (Recall/Sensitivity) no
    eixo Y para diferentes limiares de decisão. A linha diagonal
    representa o desempenho de um classificador aleatório (AUC = 0.5).
    A AUC (Area Under the Curve) é calculada e exibida na legenda.

    Args:
        y_true_bin (np.ndarray): Rótulos binarizados (1 = positiva).
        y_prob_pos (np.ndarray): Probabilidades da classe positiva.
        nome_modelo (str): Nome do modelo para o título.
        caminho_salvar (str): Caminho para salvar a imagem PNG.
    """
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob_pos)
    auc_val = roc_auc_score(y_true_bin, y_prob_pos)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC (AUC = {auc_val:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
             label="Referencia (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    plt.title(f"Curva ROC - {nome_modelo}", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curva ROC salva: {caminho_salvar}")


def gerar_classification_report_txt(
    y_true_bin, y_pred_bin, target_names, caminho_salvar
):
    """Gera e salva o classification report em arquivo de texto.

    O relatório inclui Precision, Recall, F1-Score e Suporte para cada classe,
    além da acurácia global. As classes no espaço binário são:
      - nao_ataque (classe 0): corresponde a bonafide (negativa)
      - ataque (classe 1): corresponde a attack (positiva)

    Args:
        y_true_bin (np.ndarray): Rótulos binarizados.
        y_pred_bin (np.ndarray): Predições binarizadas.
        target_names (list): Nomes das classes para o relatório.
        caminho_salvar (str): Caminho para salvar o arquivo .txt.

    Returns:
        str: O texto completo do classification report.
    """
    report = classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    acc = accuracy_score(y_true_bin, y_pred_bin)

    with open(caminho_salvar, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    print(f"  Classification report salvo: {caminho_salvar}")
    return report


def salvar_predicoes_csv(registros, caminho_salvar):
    """Salva o CSV com todas as predições individuais.

    Cada linha contém: nome do arquivo, classe real, classe predita,
    probabilidades de attack e bonafide, e indicador de acerto.

    Args:
        registros (list): Lista de dicionários com as predições.
        caminho_salvar (str): Caminho para salvar o arquivo CSV.
    """
    df = pd.DataFrame(registros)
    df.to_csv(caminho_salvar, index=False)
    print(f"  Predicoes salvas: {caminho_salvar} ({len(df)} registros)")


def salvar_erros_csv(registros, caminho_salvar):
    """Salva o CSV apenas com as classificações incorretas.

    Para cada erro, classifica o tipo:
      - FP (False Positive): bonafide classificado como attack.
      - FN (False Negative): attack classificado como bonafide.

    Args:
        registros (list): Lista de dicionários com as predições.
        caminho_salvar (str): Caminho para salvar o arquivo CSV.
    """
    erros = [r for r in registros if r["Acerto"] == False]

    if not erros:
        erros = pd.DataFrame(
            columns=[
                "Nome_Arquivo", "Classe_Real", "Classe_Predita",
                "Probabilidade_Attack", "Probabilidade_Bonafide", "Tipo_Erro"
            ]
        )
        erros.to_csv(caminho_salvar, index=False)
        print(f"  Erros salvo: {caminho_salvar} (0 erros)")
        return

    df_erros = pd.DataFrame(erros)

    def classificar_erro(row):
        if row["Classe_Real"] == "bonafide" and row["Classe_Predita"] == "attack":
            return "FP"
        elif row["Classe_Real"] == "attack" and row["Classe_Predita"] == "bonafide":
            return "FN"
        return "?"

    df_erros["Tipo_Erro"] = df_erros.apply(classificar_erro, axis=1)

    colunas = [
        "Nome_Arquivo", "Classe_Real", "Classe_Predita",
        "Probabilidade_Attack", "Probabilidade_Bonafide", "Tipo_Erro"
    ]
    df_erros = df_erros[colunas]
    df_erros.to_csv(caminho_salvar, index=False)
    print(f"  Erros salvo: {caminho_salvar} ({len(df_erros)} erros)")


def exibir_tabela_resultados(resultados):
    """Exibe a tabela resumo com as métricas de todos os modelos no terminal.

    Ordena os modelos por accuracy (decrescente) para facilitar a comparação.

    Args:
        resultados (list): Lista de dicionários com as métricas de cada modelo.
    """
    linha_sep = "=" * 140
    cabecalho = (
        f"{'MODELO':<18} {'ACC':>8} {'PREC':>8} {'REC':>8} "
        f"{'F1':>8} {'AUC':>8} {'SPEC':>8} {'HTER':>8} {'EER':>8} "
        f"{'PARAMS':>10} {'PARAM_TR':>10} {'TEMPO(ms)':>10}"
    )
    print()
    print(linha_sep)
    print(cabecalho)
    print(linha_sep)

    for r in sorted(resultados, key=lambda x: x["acc"], reverse=True):
        print(
            f"{r['modelo']:<18} {r['acc']:>8.4f} {r['prec']:>8.4f} "
            f"{r['rec']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} "
            f"{r['specificity']:>8.4f} {r['hter']:>8.4f} {r['eer']:>8.4f} "
            f"{r['params']:>10,} {r['params_treinaveis']:>10,} {r['tempo_medio']:>10.4f}"
        )

    print(linha_sep)
    print()


# =====================================================================
# SECAO 4: EXECUCAO PRINCIPAL (MAIN)
# =====================================================================


def main():
    """Função principal que orquestra todo o fluxo de inferência.

    Etapas:
      1. Localização dos arquivos .pth no diretório de modelos.
      2. Carga do dataset de teste via ImageFolder + DataLoader.
      3. Para cada modelo:
         a. Carregamento e envio para o dispositivo.
         b. Execução da inferência com medição de tempo.
         c. Cálculo das métricas principais e biométricas.
         d. Geração da matriz de confusão e curva ROC.
         e. Salvamento dos CSVs (predições, erros) e relatórios.
      4. Consolidação dos resultados em CSV comparativo.
      5. Exibição da tabela resumo final e salvamento em CSV.
      6. Listagem dos arquivos gerados.
    """
    print("=" * 70)
    print("  INFERENCIA TRANSFORMERS V2 - FACE ANTI-SPOOFING")
    print("  Rodrigo Santos - Dissertacao de Mestrado")
    print("=" * 70)
    print()
    print(f"Models dir : {MODELS_DIR}")
    print(f"Test dir   : {DATASET_DIR}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Device     : {DEVICE}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 1: Localizar arquivos .pth no diretorio de modelos
    # ------------------------------------------------------------------

    if not os.path.isdir(MODELS_DIR):
        print(f"ERRO: Diretorio de modelos nao encontrado: {MODELS_DIR}")
        sys.exit(1)

    arquivos_pth = sorted([
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".pth")
    ])

    if not arquivos_pth:
        print(f"ERRO: Nenhum arquivo .pth encontrado em {MODELS_DIR}")
        sys.exit(1)

    print(f"Modelos encontrados: {len(arquivos_pth)}")
    for pth in arquivos_pth:
        nome = os.path.basename(pth)
        print(f"  - {nome}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 2: Carregar dataset de teste
    # ------------------------------------------------------------------

    print("\n[1/5] Carregando dataset de teste...")
    try:
        loader_teste, dataset_teste = criar_dataloader_teste()
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)
    print()

    qtd_attack = sum(1 for _, cls in dataset_teste.samples if cls == 0)
    qtd_bonafide = sum(1 for _, cls in dataset_teste.samples if cls == 1)
    print(f"  Attack  : {qtd_attack}")
    print(f"  Bonafide: {qtd_bonafide}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 3: Processar cada modelo (inferencia + metricas + relatorios)
    # ------------------------------------------------------------------

    print("\n[2/5] Carregando modelos e executando inferência...")
    print()

    linhas_csv = []
    resultados_tabela = []

    for idx_pth, caminho_pth in enumerate(arquivos_pth, start=1):
        nome_arquivo = os.path.splitext(os.path.basename(caminho_pth))[0]
        nome_base = os.path.basename(caminho_pth)
        print(f"--- Modelo {idx_pth}/{len(arquivos_pth)}: {nome_arquivo} ---")

        # 3a. Carregar modelo
        try:
            model, arquitetura, info = carregar_modelo(caminho_pth)
        except Exception as e:
            print(f"  ERRO ao carregar modelo: {e}")
            print()
            continue

        print(f"  Arquitetura: {arquitetura}")
        print(f"  Device: {DEVICE}")

        total_params, trainable_params = contar_parametros(model)
        print(f"  Parametros totais: {total_params:,}")
        print(f"  Parametros treinaveis: {trainable_params:,}")

        # 3b. Inferencia
        print(f"  Executando inferencia...")
        inicio_inferencia = time.time()
        try:
            y_true, y_pred, y_prob, tempos_ms = inferir(model, loader_teste, arquitetura, inicio_inferencia)
        except Exception as e:
            print(f"  ERRO durante inferencia: {e}")
            print()
            continue

        total_imagens = len(y_true)
        tempo_total_inferencia = time.time() - inicio_inferencia
        tempo_medio_geral = tempo_total_inferencia / total_imagens if total_imagens > 0 else 0.0
        fps = 1.0 / tempo_medio_geral if tempo_medio_geral > 0 else 0.0
        th = int(tempo_total_inferencia // 3600)
        tm = int((tempo_total_inferencia % 3600) // 60)
        ts = int(tempo_total_inferencia % 60)
        print()
        print(f"Modelo conclu\u00eddo: {arquitetura}")
        print()
        print(f"Tempo total: {th:02d}:{tm:02d}:{ts:02d}")
        print()
        print(f"Imagens processadas: {_fmt_num(total_imagens)}")
        print()
        print(f"Tempo m\u00e9dio/imagem: {tempo_medio_geral:.3f} s")
        print()
        print(f"FPS equivalente: {fps:.2f}")
        print()

        # 3c. Metricas
        positive_idx = info.get("positive_idx", 0)
        acc, prec, rec, f1, auc, y_true_bin, y_pred_bin = calcular_metricas(
            y_true, y_pred, y_prob, positive_idx
        )

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  AUC      : {auc:.4f}")

        # 3d. Tempo
        tempo_stats = calcular_tempo(tempos_ms)
        print(f"  Tempo medio (ms)  : {tempo_stats['media']:.4f}")
        print(f"  Desvio padrao (ms): {tempo_stats['desvio']:.4f}")
        print(f"  Tempo min (ms)    : {tempo_stats['min']:.4f}")
        print(f"  Tempo max (ms)    : {tempo_stats['max']:.4f}")
        print(f"  Tempo total (ms)  : {tempo_stats['total']:.2f}")

        # 3e. Matriz de confusao
        cm = confusion_matrix(y_true, y_pred)
        prefixo_arquivo = nome_arquivo.replace("_final", "")
        nome_matriz = f"matriz_{prefixo_arquivo}.png"
        caminho_matriz = os.path.join(OUTPUT_DIR, nome_matriz)
        gerar_matriz_confusao(
            cm,
            classes=info["class_names"],
            titulo=f"Matriz de Confusao - {arquitetura}",
            caminho_salvar=caminho_matriz,
        )

        # 3f. Curva ROC
        nome_roc = f"roc_{prefixo_arquivo}.png"
        caminho_roc = os.path.join(OUTPUT_DIR, nome_roc)
        y_prob_pos = y_prob[:, positive_idx]
        gerar_curva_roc(y_true_bin, y_prob_pos, arquitetura, caminho_roc)

        # 3g. Metricas biometricas
        specificity, hter, eer = calcular_metricas_biometricas(cm, y_true_bin, y_prob_pos)
        print(f"  Specificity: {specificity:.4f}")
        print(f"  HTER       : {hter:.4f}")
        print(f"  EER        : {eer:.4f}")

        # 3h. Construir registros de predicoes
        assert len(dataset_teste.samples) == len(y_true), (
            f"Inconsistencia detectada: "
            f"{len(dataset_teste.samples)} amostras no dataset "
            f"e {len(y_true)} resultados de inferencia."
        )
        registros_predicoes = []
        class_names = info["class_names"]
        for idx_img, (caminho_img, _) in enumerate(dataset_teste.samples):
            nome_arquivo_img = os.path.basename(caminho_img)
            classe_real = int(y_true[idx_img])
            classe_predita = int(y_pred[idx_img])
            prob_attack = float(y_prob[idx_img, positive_idx])
            prob_bonafide = float(y_prob[idx_img, 1 - positive_idx])
            acerto = classe_real == classe_predita

            registros_predicoes.append({
                "Nome_Arquivo": nome_arquivo_img,
                "Classe_Real": class_names[classe_real],
                "Classe_Predita": class_names[classe_predita],
                "Probabilidade_Attack": prob_attack,
                "Probabilidade_Bonafide": prob_bonafide,
                "Acerto": acerto,
            })

        # 3i. CSV de predicoes
        nome_pred = f"predicoes_{prefixo_arquivo}.csv"
        caminho_pred = os.path.join(OUTPUT_DIR, nome_pred)
        salvar_predicoes_csv(registros_predicoes, caminho_pred)

        # 3j. CSV de erros
        nome_erros = f"erros_{prefixo_arquivo}.csv"
        caminho_erros = os.path.join(OUTPUT_DIR, nome_erros)
        salvar_erros_csv(registros_predicoes, caminho_erros)

        # 3k. Classification report
        nome_report = f"classification_report_{prefixo_arquivo}.txt"
        caminho_report = os.path.join(OUTPUT_DIR, nome_report)
        target_names_bin = ["nao_ataque", "ataque"]
        report_text = gerar_classification_report_txt(
            y_true_bin, y_pred_bin, target_names_bin, caminho_report
        )
        print()

        # 3l. Linha para CSV comparativo
        linhas_csv.append({
            "Modelo": arquitetura,
            "Arquitetura": ARQUITETURA_TIPO,
            "Parametros_Totais": total_params,
            "Parametros_Treinaveis": trainable_params,
            "Accuracy_Test": round(acc, 6),
            "Precision_Test": round(prec, 6),
            "Recall_Test": round(rec, 6),
            "F1_Test": round(f1, 6),
            "AUC_Test": round(auc, 6),
            "Specificity_Test": round(specificity, 6),
            "HTER_Test": round(hter, 6),
            "EER_Test": round(eer, 6),
            "Tempo_Medio_ms": round(tempo_stats["media"], 4),
            "Desvio_Padrao_ms": round(tempo_stats["desvio"], 4),
            "Inferencia_Min_ms": round(tempo_stats["min"], 4),
            "Inferencia_Max_ms": round(tempo_stats["max"], 4),
            "Tempo_Total_ms": round(tempo_stats["total"], 2),
            "Tempo_Total_s": round(tempo_total_inferencia, 2),
            "FPS": round(fps, 2),
            "Quantidade_Imagens_Test": total_imagens,
        })

        # 3m. Linha para tabela final
        resultados_tabela.append({
            "modelo": arquitetura,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
            "params": total_params,
            "params_treinaveis": trainable_params,
            "specificity": specificity,
            "hter": hter,
            "eer": eer,
            "tempo_medio": tempo_stats["media"],
            "tempo_total_s": round(tempo_total_inferencia, 2),
            "fps": round(fps, 2),
        })

        print()

        # Libera memoria do modelo
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # ETAPA 4: Salvar CSV comparativo com todas as metricas
    # ------------------------------------------------------------------

    print("\n[3/5] Salvando CSV comparativo...")

    if not linhas_csv:
        print("  Nenhum resultado para salvar.")
        sys.exit(1)

    df = pd.DataFrame(linhas_csv)
    df = df.sort_values(by="Accuracy_Test", ascending=False)
    df = df.reset_index(drop=True)

    caminho_csv = os.path.join(OUTPUT_DIR, "comparativo_test.csv")
    df.to_csv(caminho_csv, index=False)
    print(f"  CSV salvo: {caminho_csv}")
    print(f"  Linhas: {len(df)}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 5: Exibir tabela resumo final e salvar CSV de resumo
    # ------------------------------------------------------------------

    print("\n[4/5] Tabela resumo final:")
    exibir_tabela_resultados(resultados_tabela)

    # Salva resumo final em CSV
    df_resumo = pd.DataFrame([
        {
            "Modelo": r["modelo"],
            "Accuracy": round(r["acc"], 4),
            "Precision": round(r["prec"], 4),
            "Recall": round(r["rec"], 4),
            "F1_Score": round(r["f1"], 4),
            "AUC": round(r["auc"], 4),
            "Specificity": round(r["specificity"], 4),
            "HTER": round(r["hter"], 4),
            "EER": round(r["eer"], 4),
            "Parametros_Totais": r["params"],
            "Parametros_Treinaveis": r["params_treinaveis"],
            "Tempo_Medio_ms": round(r["tempo_medio"], 4),
            "Tempo_Total_s": r["tempo_total_s"],
            "FPS": r["fps"],
        }
        for r in sorted(resultados_tabela, key=lambda x: x["acc"], reverse=True)
    ])
    caminho_resumo = os.path.join(OUTPUT_DIR, "resumo_final.csv")
    df_resumo.to_csv(caminho_resumo, index=False)
    print(f"  Resumo final salvo: {caminho_resumo}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 6: Listar todos os arquivos gerados
    # ------------------------------------------------------------------

    print("\n[5/5] Arquivos gerados:")
    print(f"  {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")

    print()
    print("=" * 70)
    print("  PROCESSAMENTO CONCLUÍDO COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
