# -*- coding: utf-8 -*-
"""
Funcoes comuns aos scripts Etapa_8a (CNNs) e Etapa_8b (Transformers).

Garante que os dois scripts usem EXATAMENTE a mesma lista de imagens de
validacao (mesma ordem, mesma amostra quando AMOSTRA_POR_CLASSE for usada)
e o mesmo formato de saida dos arquivos predicoes_*.csv da Etapa 4/6.
"""
import csv
import os
import random

EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
CLASSES = ["attack", "bonafide"]          # ordem alfabetica, igual ao treino
COLUNAS = ["Nome_Arquivo", "Classe_Real", "Classe_Predita",
           "Probabilidade_Attack", "Probabilidade_Bonafide", "Acerto"]
SEMENTE_AMOSTRA = 42


def listar_imagens(pasta_subconjunto, amostra_por_classe=None):
    """Lista (caminho, classe) das imagens de um subconjunto.

    Se amostra_por_classe for um inteiro, sorteia esse numero de imagens
    por classe com semente fixa, sempre a partir da lista ordenada, de modo
    que CNNs e Transformers recebam exatamente a mesma amostra.
    """
    if not os.path.isdir(pasta_subconjunto):
        raise FileNotFoundError(f"Pasta nao encontrada: {pasta_subconjunto}")
    encontradas = sorted(d for d in os.listdir(pasta_subconjunto)
                         if os.path.isdir(os.path.join(pasta_subconjunto, d)))
    if encontradas != CLASSES:
        raise ValueError(f"Esperava as pastas {CLASSES} em {pasta_subconjunto}, "
                         f"encontrei {encontradas}")
    imagens = []
    for classe in CLASSES:
        pasta = os.path.join(pasta_subconjunto, classe)
        arquivos = sorted(f for f in os.listdir(pasta)
                          if f.lower().endswith(EXTENSOES)
                          and os.path.isfile(os.path.join(pasta, f)))
        if amostra_por_classe is not None and amostra_por_classe < len(arquivos):
            sorteio = random.Random(SEMENTE_AMOSTRA)
            arquivos = sorted(sorteio.sample(arquivos, amostra_por_classe))
        imagens.extend((os.path.join(pasta, f), classe) for f in arquivos)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem em {pasta_subconjunto}")
    return imagens


def ja_processadas(caminho_csv):
    """Nomes de arquivo ja gravados no CSV (permite retomar uma execucao)."""
    if not os.path.isfile(caminho_csv):
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {linha["Nome_Arquivo"] for linha in csv.DictReader(f)}


def abrir_csv_para_acrescentar(caminho_csv):
    novo = not os.path.isfile(caminho_csv)
    f = open(caminho_csv, "a", newline="", encoding="utf-8")
    escritor = csv.writer(f)
    if novo:
        escritor.writerow(COLUNAS)
    return f, escritor


def linha_predicao(caminho, classe_real, p_attack, p_bonafide):
    classe_pred = "attack" if p_attack >= p_bonafide else "bonafide"
    return [os.path.basename(caminho), classe_real, classe_pred,
            repr(float(p_attack)), repr(float(p_bonafide)), classe_pred == classe_real]


def formatar_tempo(segundos):
    segundos = int(segundos)
    return f"{segundos // 3600:02d}:{(segundos % 3600) // 60:02d}:{segundos % 60:02d}"
