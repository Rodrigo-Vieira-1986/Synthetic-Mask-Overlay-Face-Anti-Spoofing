# -*- coding: utf-8 -*-
"""
Etapa 8b - Inferencia dos Transformers no conjunto de VALIDACAO
================================================================

Objetivo: gerar as probabilidades de cada imagem de validacao para calibrar
o limiar do EER fora do conjunto de teste (pedido dos revisores do VISAPP).
Nenhum treinamento e feito: so os modelos ja salvos sao usados.

Carregamento e pre-processamento identicos aos da Etapa 6
(Etapa_6_Inferencia_Transformers_V2_generico.py):
  - timm.create_model(model_name, pretrained=False, num_classes) + load_state_dict
  - PIL convert("RGB") -> Resize((224, 224)) -> ToTensor -> Normalize(ImageNet)
  - softmax; probabilidade de attack pelo class_to_idx salvo no checkpoint

Saida: Resultados/Validacao_EER/predicoes_val_<modelo>.csv, no mesmo formato
dos arquivos predicoes_*.csv da Etapa 6.

A execucao pode ser interrompida e retomada: imagens ja gravadas no CSV sao
puladas na proxima execucao.

Uso (no ambiente do PyTorch):
    python Etapa_8b_Validacao_Transformers_generico.py
    python Etapa_8b_Validacao_Transformers_generico.py --modelos deit swin
    python Etapa_8b_Validacao_Transformers_generico.py --amostra 5000      (5.000 por classe)
"""
import argparse
import os
import sys
import time
import warnings

import torch
import timm
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA_SCRIPT)
from Etapa_8_comum_validacao import (listar_imagens, ja_processadas, abrir_csv_para_acrescentar,  # noqa: E402
                             linha_predicao, formatar_tempo)

# ===================== CAMINHOS (confira antes de rodar) =====================
PASTA_VALIDACAO = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET\val"
PASTA_MODELOS = os.path.join(PASTA_SCRIPT, "Resultados", "PTH")
PASTA_SAIDA = os.path.join(PASTA_SCRIPT, "Resultados", "Validacao_EER")
# ============================================================================

MODELOS = {
    "deit": {"arquivo": "deit_final.pth", "nome": "DeiT-Small"},
    "swin": {"arquivo": "swin_final.pth", "nome": "Swin-Tiny"},
    "vit":  {"arquivo": "vit_final.pth",  "nome": "ViT-B16"},
}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
TAMANHO_LOTE = 64 if DEVICE.type == "cuda" else (16 if DEVICE.type == "mps" else 8)

TRANSFORMACAO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def carregar_modelo(caminho):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["model_name"], pretrained=False, num_classes=ck.get("num_classes", 2))
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(DEVICE).eval()
    class_to_idx = ck.get("class_to_idx", {"attack": 0, "bonafide": 1})
    return model, class_to_idx


@torch.no_grad()
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
    model, class_to_idx = carregar_modelo(caminho_modelo)
    i_att, i_bon = class_to_idx["attack"], class_to_idx["bonafide"]

    arquivo, escritor = abrir_csv_para_acrescentar(caminho_csv)
    inicio = time.time()
    try:
        for i in range(0, len(faltam), TAMANHO_LOTE):
            lote = faltam[i:i + TAMANHO_LOTE]
            x = torch.stack([TRANSFORMACAO(Image.open(c).convert("RGB")) for c, _ in lote]).to(DEVICE)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            for (c, classe), p in zip(lote, probs):
                escritor.writerow(linha_predicao(c, classe, p[i_att], p[i_bon]))
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
    print(f"Dispositivo: {DEVICE} | lote: {TAMANHO_LOTE}")
    imagens = listar_imagens(a.validacao, a.amostra)
    n_att = sum(1 for _, k in imagens if k == "attack")
    print(f"Validacao: {len(imagens)} imagens ({n_att} attack, {len(imagens) - n_att} bonafide)")
    for chave in a.modelos:
        rodar_modelo(chave, imagens, a.pasta_modelos, a.saida)
    print("\nConcluido. Depois de rodar tambem a Etapa 8a, execute a Etapa 9.")


if __name__ == "__main__":
    main()
