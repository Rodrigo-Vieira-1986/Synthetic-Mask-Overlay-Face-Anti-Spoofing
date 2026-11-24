# -*- coding: utf-8 -*-
"""
Etapa 9 - Limiar do EER calibrado na validacao e aplicado ao teste
==================================================================

Le:
  - Resultados/Validacao_EER/predicoes_val_<modelo>.csv   (Etapas 8a e 8b)
  - predicoes_<modelo>.csv do TESTE, ja gerados pelas Etapas 4 e 6
    (Resultados/H5/Inferencia_CNN_V2 e Resultados/PTH/Inferencia_Transformers_V2)

Para cada modelo:
  1. Na VALIDACAO, encontra o limiar em que FAR e FRR se igualam (ponto do EER).
     Se a validacao separar as classes sem nenhum erro, usa o ponto medio entre
     a maior probabilidade de ataque de uma bona fide e a menor de um ataque.
  2. Aplica esse limiar FIXO ao TESTE (ataque se prob_attack >= limiar) e
     calcula FAR, FRR e HTER.
  3. Para comparacao, recalcula o EER com limiar escolhido no proprio teste
     (metodo da Etapa 4/6, o valor que esta no artigo).

Convencao igual a da Etapa 4/6 (classe positiva = attack):
  FAR = bona fide classificadas como ataque / total de bona fide
  FRR = ataques classificados como bona fide / total de ataques
  HTER = (FAR + FRR) / 2

Saida: Resultados/Validacao_EER/limiar_eer_calibrado.csv e tabela na tela.
Nao precisa de GPU, TensorFlow nem PyTorch.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_VAL = os.path.join(PASTA_SCRIPT, "Resultados", "Validacao_EER")
TESTE = {
    "mobilenet": ("MobileNetV2", os.path.join("H5", "Inferencia_CNN_V2", "predicoes_mobilenet.csv")),
    "resnet":    ("ResNet50",    os.path.join("H5", "Inferencia_CNN_V2", "predicoes_resnet.csv")),
    "xception":  ("Xception",    os.path.join("H5", "Inferencia_CNN_V2", "predicoes_xception.csv")),
    "deit":      ("DeiT-Small",  os.path.join("PTH", "Inferencia_Transformers_V2", "predicoes_deit.csv")),
    "swin":      ("Swin-Tiny",   os.path.join("PTH", "Inferencia_Transformers_V2", "predicoes_swin.csv")),
    "vit":       ("ViT-B/16",    os.path.join("PTH", "Inferencia_Transformers_V2", "predicoes_vit.csv")),
}


def ler(caminho):
    d = pd.read_csv(caminho)
    if d["Nome_Arquivo"].duplicated().any():
        raise ValueError(f"Arquivos repetidos em {caminho}")
    y = (d["Classe_Real"] == "attack").astype(int).to_numpy()
    s = d["Probabilidade_Attack"].astype(np.float64).to_numpy()
    return d, y, s


def limiar_eer(y, s):
    """Limiar do ponto em que FAR = FRR na amostra dada, e o EER nesse ponto."""
    max_bona = s[y == 0].max()
    min_att = s[y == 1].min()
    if max_bona < min_att:                      # separacao perfeita
        return (max_bona + min_att) / 2.0, 0.0, True
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2.0), False


def taxas(y, s, t):
    pred = (s >= t).astype(int)
    far = ((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    frr = ((pred == 0) & (y == 1)).sum() / max(1, (y == 1).sum())
    return far, frr, (far + frr) / 2.0


def eer_no_teste(y, s):
    """Mesmo calculo da Etapa 4/6 (limiar escolhido no proprio teste)."""
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta-validacao", default=PASTA_VAL)
    ap.add_argument("--pasta-resultados", default=os.path.join(PASTA_SCRIPT, "Resultados"))
    a = ap.parse_args()

    linhas, conjuntos_val = [], {}
    for chave, (nome, rel) in TESTE.items():
        cam_val = os.path.join(a.pasta_validacao, f"predicoes_val_{chave}.csv")
        cam_teste = os.path.join(a.pasta_resultados, rel)
        if not os.path.isfile(cam_val):
            print(f"[pulado] {nome}: falta {cam_val}")
            continue
        dv, yv, sv = ler(cam_val)
        dt, yt, st = ler(cam_teste)
        conjuntos_val[nome] = set(dv["Nome_Arquivo"])
        t, eer_val, perfeito = limiar_eer(yv, sv)
        far, frr, hter = taxas(yt, st, t)
        far05, frr05, hter05 = taxas(yt, st, 0.5)
        linhas.append({
            "Modelo": nome,
            "N_validacao": len(dv), "N_teste": len(dt),
            "EER_validacao_%": 100 * eer_val,
            "Limiar_validacao": t,
            "Validacao_sem_erros": perfeito,
            "Teste_FAR_%": 100 * far, "Teste_FRR_%": 100 * frr, "Teste_HTER_limiar_val_%": 100 * hter,
            "Teste_HTER_limiar_0.5_%": 100 * hter05,
            "Teste_EER_limiar_no_teste_%": 100 * eer_no_teste(yt, st),
        })

    if not linhas:
        raise SystemExit("Nenhum arquivo de validacao encontrado. Rode as Etapas 8a e 8b antes.")

    tamanhos = {len(v) for v in conjuntos_val.values()}
    if len({frozenset(v) for v in conjuntos_val.values()}) > 1:
        print("\nATENCAO: os modelos nao usaram exatamente as mesmas imagens de validacao "
              f"(tamanhos {sorted(tamanhos)}). Termine as Etapas 8a/8b com o mesmo --amostra.")

    r = pd.DataFrame(linhas)
    saida = os.path.join(a.pasta_validacao, "limiar_eer_calibrado.csv")
    r.to_csv(saida, index=False)
    pd.set_option("display.width", 200)
    print("\nLimiar do EER escolhido na validacao e aplicado ao teste (valores em %):\n")
    print(r[["Modelo", "N_validacao", "EER_validacao_%", "Limiar_validacao", "Teste_FAR_%", "Teste_FRR_%",
             "Teste_HTER_limiar_val_%", "Teste_EER_limiar_no_teste_%"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nGravado: {saida}")


if __name__ == "__main__":
    main()
