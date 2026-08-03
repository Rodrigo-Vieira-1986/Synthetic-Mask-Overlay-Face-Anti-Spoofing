"""
ETAPA_4_INFERENCIA_CNNS_V2.PY
================================================================================
Script de inferência para modelos CNN de Face Anti-Spoofing.

Descrição:
    Este script realiza a avaliação experimental completa de modelos CNN
    treinados para detecção de spoofing facial (ataque vs. bonafide).
    Carrega os modelos salvos (.h5), executa inferência no conjunto de teste,
    calcula métricas de classificação e gera relatórios e gráficos.

Arquiteturas suportadas:
    - MobileNetV2  (entrada: 224x224)
    - ResNet50     (entrada: 224x224)
    - Xception     (entrada: 299x299)

Métricas calculadas:
    Accuracy, Precision, Recall, F1-Score, ROC AUC, Specificity,
    HTER (Half Total Error Rate), EER (Equal Error Rate).

Arquivos gerados:
    - comparativo_test.csv      (métricas agregadas por modelo)
    - resumo_final.csv          (resumo ordenado por desempenho)
    - predicoes_{modelo}.csv    (predições individuais)
    - erros_{modelo}.csv        (apenas erros de classificação)
    - matriz_{modelo}.png       (matriz de confusão)
    - roc_{modelo}.png          (curva ROC)
    - classification_report_{modelo}.txt

Alterações da V2 (em relação à V1 original):
    1. Desempacotamento da matriz de confusão corrigido de
       tn, fp, fn, tp para tp, fn, fp, tn, garantindo que
       Specificity, FAR e FRR sejam calculados corretamente.
    2. Interpolação de LANCZOS para BILINEAR, correspondendo ao
       método padrão do ImageDataGenerator usado no treinamento.

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

import matplotlib
matplotlib.use("Agg")  # backend nao-interativo para ambientes sem display
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

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

# =====================================================================
# CONFIGURACOES GLOBAIS
# =====================================================================

# Suprime avisos não essenciais para manter a saída limpa durante a execução
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Diretorios dos experimentos ------------------------------------
# ================== CONFIGURE AQUI ==================
# Caminho da pasta com os modelos CNN treinados (.h5)
MODELS_DIR  = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_MODELOS_H5"
# Caminho da pasta Dataset/test com as imagens de teste
DATASET_DIR = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET_TEST"
# Caminho da pasta onde serao salvos os resultados da inferencia CNN
OUTPUT_DIR  = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_RESULTADOS_CNN"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Semente aleatoria para reproducibilidade -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Variaveis globais (preenchidas durante a execucao) -------------
CLASS_NAMES  = None   # lista de nomes das classes (ex.: ["attack", "bonafide"])
CLASS_TO_IDX = None   # dicionario: nome da classe -> indice numerico
POSITIVE_CLASS = None # indice da classe positiva (attack)

# --- Mapeamento: nome do arquivo .h5 -> metadados da arquitetura -----------
ARQUITETURAS = {
    "mobilenet_final.h5": {"nome": "MobileNetV2", "input_size": 224},
    "resnet_final.h5":    {"nome": "ResNet50",    "input_size": 224},
    "xception_final.h5":  {"nome": "Xception",    "input_size": 299},
}

ARQUITETURA_TIPO = "CNN"


# =====================================================================
# FUNCOES AUXILIARES DE FORMATACAO E PROGRESSO
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


# =====================================================================
# SECAO 1: FUNCOES DE VALIDACAO DO DATASET E PRE-PROCESSAMENTO
# =====================================================================

def obter_classes_do_dataset():
    """Descobre e valida as classes a partir dos subdiretórios do dataset de teste.

    O dataset de teste deve conter exatamente duas subpastas: 'attack' e 'bonafide'.
    A ordenação alfabética define o índice numérico de cada classe, que deve
    corresponder à ordenação utilizada durante o treinamento.

    Returns:
        tuple: (class_names, class_to_idx)
            - class_names (list): Lista ordenada dos nomes das classes.
            - class_to_idx (dict): Mapeamento nome -> índice.

    Raises:
        FileNotFoundError: Se o diretório de teste não existir.
        ValueError: Se alguma classe esperada estiver ausente ou se houver
                    classes inesperadas.
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Diretorio de teste nao encontrado: {DATASET_DIR}")

    # Lista as subpastas e ordena alfabeticamente (mesmo comportamento do
    # ImageDataGenerator.flow_from_directory no treinamento)
    classes_encontradas = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    classes_esperadas = ["attack", "bonafide"]

    # Verifica se todas as classes esperadas estao presentes
    for cls in classes_esperadas:
        if cls not in classes_encontradas:
            raise ValueError(
                f"Classe esperada '{cls}' nao encontrada em {DATASET_DIR}.\n"
                f"Classes encontradas: {classes_encontradas}"
            )

    # Rejeita classes inesperadas (protecao contra contaminacao do dataset)
    for cls in classes_encontradas:
        if cls not in classes_esperadas:
            raise ValueError(
                f"Classe inesperada '{cls}' encontrada em {DATASET_DIR}.\n"
                f"Classes esperadas: {classes_esperadas}"
            )

    # Cria o mapeamento: 'attack' -> 0, 'bonafide' -> 1
    class_to_idx = {cls: idx for idx, cls in enumerate(classes_encontradas)}
    return classes_encontradas, class_to_idx


def preprocess_imagem(caminho_imagem, input_size, arquitetura):
    """Carrega e pré-processa uma imagem para inferência.

    O pipeline reproduz exatamente o pré-processamento aplicado durante o
    treinamento (Etapa_3):
      1. Carregamento da imagem em RGB.
      2. Redimensionamento com interpolação BILINEAR (padrão do
         ImageDataGenerator usado no treino, TF 2.15).
      3. Conversão para array float32.
      4. Aplicação da função preprocess_input específica da arquitetura
         (mobilenet_preprocess, resnet_preprocess ou xception_preprocess).

    Args:
        caminho_imagem (str): Caminho absoluto para o arquivo de imagem.
        input_size (int): Tamanho alvo (largura = altura) para redimensionamento.
        arquitetura (str): Nome da arquitetura ("MobileNetV2", "ResNet50", "Xception").

    Returns:
        np.ndarray: Array 3D (height, width, 3) com a imagem pré-processada,
                    pronta para ser passada ao modelo.

    Raises:
        ValueError: Se houver erro no carregamento ou processamento da imagem.
    """
    try:
        # Passo 1: Carregamento e conversao para RGB
        img = Image.open(caminho_imagem).convert("RGB")

        # Passo 2: Redimensionamento com BILINEAR (V2: corrigido de LANCZOS
        # para BILINEAR, correspondendo ao metodo padrao do ImageDataGenerator
        # utilizado no treinamento com TensorFlow 2.15)
        img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)

        # Passo 3: Conversao para array float32
        img_array = np.array(img, dtype=np.float32)

        # Passo 4: Aplicacao do preprocess_input especifico de cada arquitetura
        # As funcoes preprocess_input foram utilizadas no treinamento como
        # preprocessing_function do ImageDataGenerator e devem ser replicadas
        # na inferencia para garantir consistencia nos valores de entrada.
        if arquitetura == "MobileNetV2":
            # MobileNetV2: escala de [0,255] para [-1,1]
            img_array = mobilenet_preprocess(img_array)
        elif arquitetura == "ResNet50":
            # ResNet50: normalizacao com media e desvio padrao do ImageNet
            img_array = resnet_preprocess(img_array)
        elif arquitetura == "Xception":
            # Xception: escala de [0,255] para [-1,1]
            img_array = xception_preprocess(img_array)

        return img_array

    except Exception as e:
        raise ValueError(f"Erro ao carregar imagem {caminho_imagem}: {e}")


def listar_imagens_teste():
    """Percorre os subdiretórios de teste e monta a lista de imagens.

    Para cada classe, lista os arquivos de imagem com extensões suportadas
    (.png, .jpg, .jpeg, .bmp, .tiff, .tif) e retorna uma lista de tuplas
    (caminho_absoluto, índice_da_classe).

    Returns:
        list: Lista de tuplas (caminho_absoluto, classe_idx).

    Raises:
        FileNotFoundError: Se o diretório de teste não existir ou se
                           nenhuma imagem for encontrada.
    """
    imagens = []
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Diretorio de teste nao encontrado: {DATASET_DIR}")

    for classe_nome, classe_idx in CLASS_TO_IDX.items():
        classe_dir = os.path.join(DATASET_DIR, classe_nome)
        if not os.path.isdir(classe_dir):
            print(f"  Aviso: diretorio {classe_dir} nao encontrado")
            continue

        for arquivo in sorted(os.listdir(classe_dir)):
            caminho = os.path.join(classe_dir, arquivo)
            if os.path.isfile(caminho) and arquivo.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
            ):
                imagens.append((caminho, classe_idx))

    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {DATASET_DIR}")

    return imagens


def extrair_nome_arquivo(caminho):
    """Extrai apenas o nome do arquivo (com extensao) de um caminho completo.

    Args:
        caminho (str): Caminho completo do arquivo.

    Returns:
        str: Nome do arquivo com extensao.
    """
    return os.path.basename(caminho)


# =====================================================================
# SECAO 2: FUNCOES DE IDENTIFICACAO E CONTAGEM DO MODELO
# =====================================================================

def identificar_arquitetura_do_modelo(model):
    """Identifica a arquitetura da CNN analisando os nomes das camadas.

    Percorre as camadas do modelo carregado em busca de padrões nos nomes
    que indiquem a arquitetura base utilizada durante o treinamento.

    Args:
        model (tf.keras.Model): Modelo carregado.

    Returns:
        str or None: Nome da arquitetura ("MobileNetV2", "ResNet50", "Xception")
                     ou None se não for possível identificar.
    """
    for layer in model.layers:
        name_lower = layer.name.lower()
        if 'mobilenetv2' in name_lower:
            return "MobileNetV2"
        if 'xception' in name_lower:
            return "Xception"

    for layer in model.layers:
        name_lower = layer.name.lower()
        if name_lower.startswith('res') and len(name_lower) > 3 and name_lower[3].isdigit():
            return "ResNet50"

    return None


def identificar_arquitetura(nome_arquivo, model=None):
    """Identifica a arquitetura do modelo por dois métodos em cascata.

    Método 1 (prioritário): Analisa as camadas do modelo carregado.
    Método 2 (fallback): Usa o nome do arquivo .h5 para inferir a arquitetura.

    Args:
        nome_arquivo (str): Nome do arquivo .h5.
        model (tf.keras.Model, optional): Modelo carregado. Se fornecido,
                                           a identificação por camadas tem prioridade.

    Returns:
        tuple: (nome_arquitetura, input_size) ou apenas nome_arquitetura
               quando a identificação é feita pelo modelo.
    """
    if model is not None:
        arq = identificar_arquitetura_do_modelo(model)
        if arq is not None:
            return arq

    nome_base = os.path.basename(nome_arquivo)
    if nome_base in ARQUITETURAS:
        info = ARQUITETURAS[nome_base]
        return info["nome"], info["input_size"]

    # Fallback por prefixo no nome do arquivo
    nome_lower = nome_base.lower()
    if "mobilenet" in nome_lower:
        return "MobileNetV2", 224
    elif "resnet" in nome_lower:
        return "ResNet50", 224
    elif "xception" in nome_lower:
        return "Xception", 299

    return "Desconhecido", 224


def contar_parametros(model):
    """Conta os parâmetros totais e treináveis do modelo.

    Args:
        model (tf.keras.Model): Modelo carregado.

    Returns:
        tuple: (total_params, trainable_params)
            - total_params (int): Número total de parâmetros.
            - trainable_params (int): Número de parâmetros treináveis.
    """
    total = model.count_params()
    treinaveis = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    return total, treinaveis


# =====================================================================
# SECAO 3: FUNCOES DE CALCULO DE METRICAS
# =====================================================================

def calcular_metricas(y_true, y_pred, y_prob, positive_class=0):
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
        positive_class (int): Índice da classe positiva (default: 0 = attack).

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
    # Probabilidade da classe positiva (usada para ROC AUC)
    y_prob_pos = y_prob[:, positive_class]

    # Binarizacao: 1 se for a classe positiva (attack), 0 caso contrario
    y_true_bin = np.where(y_true == positive_class, 1, 0)
    y_pred_bin = np.where(y_pred == positive_class, 1, 0)

    # --- Accuracy: proporcao de acertos no total de amostras -------------
    acc = accuracy_score(y_true, y_pred)

    # --- Precision: TP / (TP + FP) ------------------------------------
    # Mede quantas das deteccoes de ataque sao realmente ataques.
    # No contexto de Face Anti-Spoofing, uma precision baixa significa
    # que muitos usuarios bonafide estao sendo incorretamente rejeitados.
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)

    # --- Recall (TPR): TP / (TP + FN) ----------------------------------
    # Mede quantos dos ataques reais foram detectados.
    # No contexto de Face Anti-Spoofing, um recall baixo significa que
    # muitos ataques estao passando pelo sistema.
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    # --- F1-Score: media harmonica entre Precision e Recall -----------
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # --- ROC AUC: Area sob a Curva ROC --------------------------------
    # Mede a capacidade do modelo de separar as classes independentemente
    # do limiar de decisao. AUC = 1 indica separacao perfeita; AUC = 0.5
    # indica desempenho aleatorio.
    try:
        auc = roc_auc_score(y_true_bin, y_prob_pos)
    except ValueError:
        # Pode ocorrer se houver apenas uma classe nos dados
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
    # V2: Desempacotamento corrigido da matriz de confusao.
    # sklearn confusion_matrix(labels=[0, 1]) retorna [[TP, FN], [FP, TN]].
    # Portanto cm.ravel() = [tp, fn, fp, tn] para labels [0=attack, 1=bonafide].
    tp, fn, fp, tn = cm.ravel()

    # --- Specificity (TNR) = TN / (TN + FP) ---------------------------
    denom_spec = tn + fp
    specificity = tn / denom_spec if denom_spec > 0 else 0.0

    # --- FAR = FP / (FP + TN) -----------------------------------------
    denom_far = fp + tn
    far = fp / denom_far if denom_far > 0 else 0.0

    # --- FRR = FN / (FN + TP) -----------------------------------------
    denom_frr = fn + tp
    frr = fn / denom_frr if denom_frr > 0 else 0.0

    # --- HTER = (FAR + FRR) / 2 ---------------------------------------
    hter = (far + frr) / 2.0

    # --- EER: ponto onde FPR = FNR ------------------------------------
    # O EER e encontrado variando o limiar de decisao e identificando
    # o ponto onde a taxa de falsos positivos (FPR) se iguala a taxa de
    # falsos negativos (FNR = 1 - TPR).
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_pos)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
    except Exception:
        eer = 0.0

    return specificity, hter, eer


# =====================================================================
# SECAO 4: FUNCOES DE ESTATISTICAS DE TEMPO
# =====================================================================

def calcular_estatisticas_tempo(tempos_ms):
    """Calcula estatísticas descritivas dos tempos de inferência.

    Args:
        tempos_ms (np.ndarray): Array com os tempos de inferência em
                                milissegundos para cada imagem.

    Returns:
        dict: Dicionário com média, desvio, mínimo, máximo e total dos tempos.
    """
    if len(tempos_ms) == 0:
        return {"media": 0, "desvio": 0, "min": 0, "max": 0, "total": 0}

    return {
        "media":  float(np.mean(tempos_ms)),
        "desvio": float(np.std(tempos_ms)),
        "min":    float(np.min(tempos_ms)),
        "max":    float(np.max(tempos_ms)),
        "total":  float(np.sum(tempos_ms)),
    }


# =====================================================================
# SECAO 5: FUNCOES DE GERACAO DE GRAFICOS E RELATORIOS
# =====================================================================

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


def gerar_classification_report_txt(y_true_bin, y_pred_bin, target_names, caminho_salvar):
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


# =====================================================================
# SECAO 6: FUNCOES DE SALVAMENTO DE CSVs
# =====================================================================

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


# =====================================================================
# SECAO 7: FUNCAO DE EXIBICAO DA TABELA DE RESULTADOS
# =====================================================================

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
# SECAO 8: FUNCAO PRINCIPAL (MAIN)
# =====================================================================

def main():
    """Função principal que orquestra todo o fluxo de inferência.

    Etapas:
      1. Localização e filtragem dos arquivos .h5 no diretório de modelos.
      2. Validação do dataset e descoberta das classes (attack, bonafide).
      3. Listagem das imagens de teste.
      4. Para cada modelo:
         a. Carregamento e validação da arquitetura.
         b. Execução da inferência com medição de tempo.
         c. Cálculo das métricas de classificação e biométricas.
         d. Geração da matriz de confusão e curva ROC.
         e. Salvamento dos CSVs (predições, erros) e relatórios.
      5. Consolidação dos resultados em CSV comparativo.
      6. Exibição da tabela resumo final e salvamento do resumo.
    """
    print("=" * 70)
    print("  INFERENCIA CNNs V2 - FACE ANTI-SPOOFING")
    print("  Rodrigo Santos - Dissertacao de Mestrado")
    print("=" * 70)
    print()
    print(f"Models dir : {MODELS_DIR}")
    print(f"Test dir   : {DATASET_DIR}")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 1: Localizar arquivos .h5 no diretorio de modelos
    # ------------------------------------------------------------------
    if not os.path.isdir(MODELS_DIR):
        print(f"ERRO: Diretorio de modelos nao encontrado: {MODELS_DIR}")
        sys.exit(1)

    arquivos_h5 = sorted([
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".h5")
    ])

    if not arquivos_h5:
        print(f"ERRO: Nenhum arquivo .h5 encontrado em {MODELS_DIR}")
        sys.exit(1)

    # Filtra apenas os modelos esperados (MobileNetV2, ResNet50, Xception)
    modelos_esperados = ["mobilenet_final.h5", "resnet_final.h5", "xception_final.h5"]
    arquivos_h5 = [
        p for p in arquivos_h5
        if os.path.basename(p) in modelos_esperados
    ]

    if not arquivos_h5:
        print(
            f"ERRO: Nenhum modelo esperado encontrado em {MODELS_DIR}.\n"
            f"Esperado: {modelos_esperados}"
        )
        sys.exit(1)

    print(f"Modelos encontrados: {len(arquivos_h5)}")
    for pth in arquivos_h5:
        nome = os.path.basename(pth)
        arquitetura, _ = identificar_arquitetura(nome)
        print(f"  - {nome} ({arquitetura})")
    print()

    # ------------------------------------------------------------------
    # ETAPA 2: Validar dataset e descobrir as classes
    # ------------------------------------------------------------------
    print("[1/5] Validando dataset e descobrindo classes...")
    try:
        global CLASS_NAMES, CLASS_TO_IDX, POSITIVE_CLASS
        CLASS_NAMES, CLASS_TO_IDX = obter_classes_do_dataset()
        # A classe 'attack' e definida como positiva (deteccao de spoofing).
        # A classe positiva e aquela que se deseja detectar: ataques faciais.
        POSITIVE_CLASS = CLASS_TO_IDX["attack"]
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    print(f"  Classes encontradas: {CLASS_NAMES}")
    print(f"  Mapeamento: {CLASS_TO_IDX}")
    print(f"  Classe positiva (attack): indice {POSITIVE_CLASS}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 3: Listar imagens de teste
    # ------------------------------------------------------------------
    print("[2/5] Listando imagens de teste...")
    try:
        imagens_teste = listar_imagens_teste()
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    print(f"  Total de imagens: {len(imagens_teste)}")
    qtd_attack = sum(1 for _, cls in imagens_teste if cls == CLASS_TO_IDX["attack"])
    qtd_bonafide = sum(1 for _, cls in imagens_teste if cls == CLASS_TO_IDX["bonafide"])
    print(f"  Attack  : {qtd_attack}")
    print(f"  Bonafide: {qtd_bonafide}")
    print()

    # ------------------------------------------------------------------
    # ETAPA 4: Processar cada modelo
    # ------------------------------------------------------------------
    print("[3/5] Carregando modelos e executando inferencia...")
    print()

    linhas_csv = []          # Linhas para o CSV comparativo
    resultados_tabela = []   # Resultados para a tabela final

    for idx_h5, caminho_h5 in enumerate(arquivos_h5, start=1):
        nome_arquivo = os.path.splitext(os.path.basename(caminho_h5))[0]
        nome_base = os.path.basename(caminho_h5)

        # Prefixo para nomeacao dos arquivos de saida (ex.: "mobilenet", "resnet")
        prefixo = nome_arquivo.replace("_final", "")

        # --- 4a. Carregar modelo ----------------------------------------
        print(f"--- Modelo {idx_h5}/{len(arquivos_h5)}: {nome_base} ---")
        print(f"  Carregando {nome_base}...")
        try:
            model = tf.keras.models.load_model(caminho_h5, compile=False)
        except Exception as e:
            print(f"  ERRO ao carregar modelo {caminho_h5}: {e}")
            print()
            continue

        # Identifica a arquitetura (prioritariamente pelas camadas do modelo)
        arquitetura = identificar_arquitetura_do_modelo(model)
        if arquitetura is None:
            arquitetura, _ = identificar_arquitetura(nome_base)

        # Obtem o tamanho de entrada diretamente do modelo carregado
        input_size = model.input_shape[1]

        # Valida o numero de classes na camada de saida
        n_classes_saida = model.output_shape[-1]
        if n_classes_saida != 2:
            print(f"  ERRO: Modelo {nome_base} possui {n_classes_saida} classes na saida.")
            print(f"  Esperado: 2 classes. Incompatibilidade entre treinamento e inferencia.")
            del model
            tf.keras.backend.clear_session()
            continue

        print(f"  Arquitetura: {arquitetura}")
        print(f"  Input size : {input_size}x{input_size}")

        total_params, trainable_params = contar_parametros(model)
        print(f"  Parametros totais    : {total_params:,}")
        print(f"  Parametros treinaveis: {trainable_params:,}")

        # --- 4b. Executar inferencia em todas as imagens ------------------
        print(f"  Executando inferencia em {len(imagens_teste)} imagens...")

        # Warm-up: uma predicao com tensor zero para aquecer o modelo
        _ = model.predict(
            np.zeros((1, input_size, input_size, 3), dtype=np.float32),
            verbose=0,
        )

        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        tempos_ms = []
        registros_predicoes = []

        total = len(imagens_teste)
        inicio_inferencia = time.time()

        for idx_img, (caminho_img, classe_real) in enumerate(imagens_teste, start=1):
            try:
                img_array = preprocess_imagem(caminho_img, input_size, arquitetura)
                img_batch = np.expand_dims(img_array, axis=0)

                inicio = time.perf_counter()
                prob = model.predict(img_batch, verbose=0)[0]
                fim = time.perf_counter()
            except Exception as e:
                print(f"  Aviso: erro ao processar {caminho_img}: {e}")
                continue

            tempo_ms = (fim - inicio) * 1000.0
            tempos_ms.append(tempo_ms)

            classe_predita = int(np.argmax(prob))
            prob_attack = float(prob[CLASS_TO_IDX["attack"]])
            prob_bonafide = float(prob[CLASS_TO_IDX["bonafide"]])

            y_true_list.append(classe_real)
            y_pred_list.append(classe_predita)
            y_prob_list.append(prob)

            nome_arquivo_img = extrair_nome_arquivo(caminho_img)
            acerto = classe_real == classe_predita

            registros_predicoes.append({
                "Nome_Arquivo": nome_arquivo_img,
                "Classe_Real": CLASS_NAMES[classe_real],
                "Classe_Predita": CLASS_NAMES[classe_predita],
                "Probabilidade_Attack": prob_attack,
                "Probabilidade_Bonafide": prob_bonafide,
                "Acerto": acerto,
            })

            if idx_img % 100 == 0 or idx_img == total:
                tempo_decorrido = time.time() - inicio_inferencia
                tempo_por_img = tempo_decorrido / idx_img
                linha = _formatar_linha_progresso(
                    arquitetura, idx_img, total, tempo_decorrido, tempo_por_img
                )
                print(linha.ljust(130), end="\r", flush=True)

        print()
        tempo_total_inferencia = time.time() - inicio_inferencia
        tempo_medio_geral = tempo_total_inferencia / total
        fps = 1.0 / tempo_medio_geral if tempo_medio_geral > 0 else 0.0
        th = int(tempo_total_inferencia // 3600)
        tm = int((tempo_total_inferencia % 3600) // 60)
        ts = int(tempo_total_inferencia % 60)
        print(f"Modelo concluido: {arquitetura}")
        print()
        print(f"Tempo total: {th:02d}:{tm:02d}:{ts:02d}")
        print()
        print(f"Imagens processadas: {_fmt_num(total)}")
        print()
        print(f"Tempo medio/imagem: {tempo_medio_geral:.3f} s")
        print()
        print(f"FPS equivalente: {fps:.2f}")
        print()

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)
        tempos_arr = np.array(tempos_ms)

        # --- 4c. Calcular metricas ---------------------------------------
        print(f"  Calculando metricas...")
        acc, prec, rec, f1, auc, y_true_bin, y_pred_bin = calcular_metricas(
            y_true, y_pred, y_prob, positive_class=POSITIVE_CLASS
        )

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  AUC      : {auc:.4f}")

        # --- 4d. Estatisticas de tempo -----------------------------------
        tempo_stats = calcular_estatisticas_tempo(tempos_arr)
        print(f"  Tempo medio (ms)  : {tempo_stats['media']:.4f}")
        print(f"  Desvio padrao (ms): {tempo_stats['desvio']:.4f}")
        print(f"  Tempo min (ms)    : {tempo_stats['min']:.4f}")
        print(f"  Tempo max (ms)    : {tempo_stats['max']:.4f}")
        print(f"  Tempo total (ms)  : {tempo_stats['total']:.2f}")

        # --- 4e. Matriz de confusao --------------------------------------
        cm = confusion_matrix(y_true, y_pred)
        nome_matriz = f"matriz_{prefixo}.png"
        caminho_matriz = os.path.join(OUTPUT_DIR, nome_matriz)
        gerar_matriz_confusao(
            cm,
            classes=CLASS_NAMES,
            titulo=f"Matriz de Confusao - {arquitetura}",
            caminho_salvar=caminho_matriz,
        )

        # --- 4f. Curva ROC -----------------------------------------------
        nome_roc = f"roc_{prefixo}.png"
        caminho_roc = os.path.join(OUTPUT_DIR, nome_roc)
        y_prob_pos = y_prob[:, POSITIVE_CLASS]
        gerar_curva_roc(y_true_bin, y_prob_pos, arquitetura, caminho_roc)

        # --- 4g. Metricas biometricas ------------------------------------
        specificity, hter, eer = calcular_metricas_biometricas(
            cm, y_true_bin, y_prob_pos
        )
        print(f"  Specificity: {specificity:.4f}")
        print(f"  HTER       : {hter:.4f}")
        print(f"  EER        : {eer:.4f}")

        # --- 4h. Classification report -----------------------------------
        nome_report = f"classification_report_{prefixo}.txt"
        caminho_report = os.path.join(OUTPUT_DIR, nome_report)
        target_names_bin = ["nao_ataque", "ataque"]
        gerar_classification_report_txt(
            y_true_bin, y_pred_bin, target_names_bin, caminho_report
        )

        # --- 4i. CSV de predicoes ----------------------------------------
        nome_pred = f"predicoes_{prefixo}.csv"
        caminho_pred = os.path.join(OUTPUT_DIR, nome_pred)
        salvar_predicoes_csv(registros_predicoes, caminho_pred)

        # --- 4j. CSV de erros --------------------------------------------
        nome_erros = f"erros_{prefixo}.csv"
        caminho_erros = os.path.join(OUTPUT_DIR, nome_erros)
        salvar_erros_csv(registros_predicoes, caminho_erros)

        # --- 4k. Linha para CSV comparativo ------------------------------
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
            "Quantidade_Imagens_Test": len(y_true),
        })

        # --- 4l. Linha para tabela final ---------------------------------
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
        })

        print()

        # Libera memoria do modelo
        del model
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # ETAPA 5: Salvar CSV comparativo
    # ------------------------------------------------------------------
    print("[4/5] Salvando CSV comparativo...")

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
    # ETAPA 6: Exibir tabela resumo final e salvar CSV de resumo
    # ------------------------------------------------------------------
    print("[5/5] Tabela resumo final:")
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
        }
        for r in sorted(resultados_tabela, key=lambda x: x["acc"], reverse=True)
    ])
    caminho_resumo = os.path.join(OUTPUT_DIR, "resumo_final.csv")
    df_resumo.to_csv(caminho_resumo, index=False)
    print(f"  Resumo final salvo: {caminho_resumo}")
    print()

    print("  Arquivos gerados:")
    print(f"  {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")

    print()
    print("=" * 70)
    print("  PROCESSAMENTO CONCLUIDO COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
