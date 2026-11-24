# -*- coding: utf-8 -*-
"""
Etapa 8a - Inferencia das CNNs no conjunto de VALIDACAO
=========================================================

Objetivo: gerar as probabilidades de cada imagem de validacao para calibrar
o limiar do EER fora do conjunto de teste (pedido dos revisores do VISAPP).
Nenhum treinamento e feito: so os modelos ja salvos sao usados.

O pre-processamento e identico ao da Etapa 4 (Etapa_4_Inferencia_CNNs_V2_generico.py):
  1. PIL Image.open(...).convert("RGB")
  2. resize para input_size com interpolacao BILINEAR
  3. float32
  4. preprocess_input especifico da arquitetura

Unica diferenca: as imagens sao processadas em lotes (mais rapido). O
resultado por imagem e o mesmo.

Saida: Resultados/Validacao_EER/predicoes_val_<modelo>.csv, no mesmo formato
dos arquivos predicoes_*.csv da Etapa 4.

A execucao pode ser interrompida e retomada: imagens ja gravadas no CSV sao
puladas na proxima execucao.

Uso (no ambiente do TensorFlow):
    python Etapa_8a_Validacao_CNNs_generico.py
    python Etapa_8a_Validacao_CNNs_generico.py --modelos resnet xception
    python Etapa_8a_Validacao_CNNs_generico.py --amostra 5000      (5.000 por classe)
"""
import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess  # noqa: E402
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess  # noqa: E402
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess  # noqa: E402

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA_SCRIPT)
from Etapa_8_comum_validacao import (listar_imagens, ja_processadas, abrir_csv_para_acrescentar,  # noqa: E402
                             linha_predicao, formatar_tempo)

# ===================== CAMINHOS (confira antes de rodar) =====================
PASTA_VALIDACAO = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET\val"
PASTA_MODELOS = os.path.join(PASTA_SCRIPT, "Resultados", "H5")
PASTA_SAIDA = os.path.join(PASTA_SCRIPT, "Resultados", "Validacao_EER")
# ============================================================================

MODELOS = {
    "mobilenet": {"arquivo": "mobilenet_final.h5", "nome": "MobileNetV2", "input_size": 224, "prep": mobilenet_preprocess},
    "resnet":    {"arquivo": "resnet_final.h5",    "nome": "ResNet50",    "input_size": 224, "prep": resnet_preprocess},
    "xception":  {"arquivo": "xception_final.h5",  "nome": "Xception",    "input_size": 299, "prep": xception_preprocess},
}
CLASS_TO_IDX = {"attack": 0, "bonafide": 1}   # mesmo mapeamento da Etapa 3 e 4
TAMANHO_LOTE = 32


def _camada_compativel(base, ignorar):
    """Subclasse da camada que descarta parametros salvos por versoes antigas do Keras."""
    class Compativel(base):
        _ignorar = frozenset(ignorar)

        def __init__(self, *args, **kwargs):
            for chave in self._ignorar:
                kwargs.pop(chave, None)
            super().__init__(*args, **kwargs)
    Compativel.__name__ = base.__name__
    return Compativel


def carregar_modelo(caminho, keras_mod=None):
    """Abre o .h5 salvo no TensorFlow 2.15 (Keras 2) em qualquer Keras 3.

    Versoes do Keras 3 anteriores a ~3.8 recusam parametros antigos como
    renorm, renorm_clipping e renorm_momentum (BatchNormalization). Eles nao
    afetam a inferencia (renorm=False e o padrao); aqui sao descartados
    automaticamente, camada por camada, ate o modelo abrir.
    """
    keras_mod = keras_mod or tf.keras
    extras = {}
    for _ in range(50):
        try:
            return keras_mod.models.load_model(caminho, compile=False, custom_objects=extras)
        except Exception as erro:
            m = re.search(r"Unrecognized keyword arguments? passed to (\w+): (\{.*?\})", str(erro))
            if not m:
                raise
            classe = m.group(1)
            chaves = set(re.findall(r"'(\w+)'\s*:", m.group(2)))
            anteriores = getattr(extras.get(classe), "_ignorar", frozenset())
            if chaves <= anteriores:
                raise
            extras[classe] = _camada_compativel(getattr(keras_mod.layers, classe), anteriores | chaves)
            print(f"  compatibilidade: ignorando {sorted(chaves)} em {classe}")
    raise RuntimeError("Nao foi possivel abrir o modelo " + caminho)


def preparar(caminho, input_size, prep):
    img = Image.open(caminho).convert("RGB")
    img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    return prep(np.array(img, dtype=np.float32))


def rodar_modelo(chave, imagens, pasta_modelos, pasta_saida):
    cfg = MODELOS[chave]
    caminho_modelo = os.path.join(pasta_modelos, cfg["arquivo"])
    caminho_csv = os.path.join(pasta_saida, f"predicoes_val_{chave}.csv")
    feitas = ja_processadas(caminho_csv)
    faltam = [(c, k) for c, k in imagens if os.path.basename(c) not in feitas]
    print(f"\n{cfg['nome']}: {len(imagens)} imagens, {len(feitas)} ja feitas, {len(faltam)} a fazer")
    if not faltam:
        return
    if not os.path.isfile(caminho_modelo):
        raise FileNotFoundError(f"Modelo nao encontrado: {caminho_modelo}")
    model = carregar_modelo(caminho_modelo)

    arquivo, escritor = abrir_csv_para_acrescentar(caminho_csv)
    inicio = time.time()
    try:
        for i in range(0, len(faltam), TAMANHO_LOTE):
            lote = faltam[i:i + TAMANHO_LOTE]
            x = np.stack([preparar(c, cfg["input_size"], cfg["prep"]) for c, _ in lote])
            probs = np.asarray(model.predict_on_batch(x))
            for (c, classe), p in zip(lote, probs):
                escritor.writerow(linha_predicao(c, classe,
                                                 p[CLASS_TO_IDX["attack"]], p[CLASS_TO_IDX["bonafide"]]))
            arquivo.flush()
            feitas_agora = i + len(lote)
            decorrido = time.time() - inicio
            eta = decorrido / feitas_agora * (len(faltam) - feitas_agora)
            print(f"  {cfg['nome']}: {feitas_agora}/{len(faltam)} | tempo {formatar_tempo(decorrido)} "
                  f"| faltam {formatar_tempo(eta)}", end="\r", flush=True)
    finally:
        arquivo.close()
    print(f"\n  gravado: {caminho_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    ap.add_argument("--amostra", type=int, default=None,
                    help="imagens por classe (padrao: validacao inteira)")
    ap.add_argument("--validacao", default=PASTA_VALIDACAO)
    ap.add_argument("--pasta-modelos", default=PASTA_MODELOS)
    ap.add_argument("--saida", default=PASTA_SAIDA)
    a = ap.parse_args()

    os.makedirs(a.saida, exist_ok=True)
    imagens = listar_imagens(a.validacao, a.amostra)
    n_att = sum(1 for _, k in imagens if k == "attack")
    print(f"Validacao: {len(imagens)} imagens ({n_att} attack, {len(imagens) - n_att} bonafide)")
    for chave in a.modelos:
        rodar_modelo(chave, imagens, a.pasta_modelos, a.saida)
    print("\nConcluido. Depois de rodar tambem a Etapa 8b, execute a Etapa 9.")


if __name__ == "__main__":
    main()
