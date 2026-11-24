# -*- coding: utf-8 -*-
"""
Etapa 7 - Exporta a particao treino/validacao/teste usada nos experimentos
===========================================================================

O script de divisao (Etapa_2_Dataset.py) nao fixa semente aleatoria, entao a
particao usada no artigo so existe nas pastas Dataset/{train,val,test}. Este
script grava essa particao em um CSV, para publicacao junto ao codigo e as
anotacoes, permitindo que outros pesquisadores reproduzam os resultados.

So le nomes de arquivo. Nao copia, move nem altera nenhuma imagem.

Colunas do CSV:
  subconjunto     train, val ou test
  classe          attack ou bonafide
  arquivo         nome do arquivo (ex.: attack_000002.jpg, bonafide_00008.jpg)
  caminho_relativo  caminho a partir da pasta Dataset
  indice_celeba   numero da imagem de origem no CelebA, com 6 digitos
                  (attack_000002 -> 000002; bonafide_00008 -> 000008)

O indice permite a quem tiver o CelebA cruzar com identity_CelebA.txt. As
identidades nao sao gravadas aqui, por serem anotacoes do proprio CelebA.

Uso:
    python Etapa_7_Exportar_Particao_generico.py
"""
import argparse
import csv
import os
import re
from collections import Counter

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_DATASET = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET"
ARQUIVO_SAIDA = os.path.join(PASTA_SCRIPT, "particao_experimentos.csv")

SUBCONJUNTOS = ["train", "val", "test"]
CLASSES = ["attack", "bonafide"]
EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
MAX_CELEBA = 202599

# Contagens publicadas no artigo (Tabela 2), usadas so para conferencia
ESPERADO = {
    ("train", "attack"): 141444, ("train", "bonafide"): 141819,
    ("val", "attack"): 40412, ("val", "bonafide"): 40519,
    ("test", "attack"): 20207, ("test", "bonafide"): 20261,
}


def indice_celeba(nome):
    """Numero da imagem do CelebA a partir do nome, com ou sem zeros a esquerda."""
    candidatos = [g for g in re.findall(r"\d+", os.path.splitext(nome)[0]) if 1 <= int(g) <= MAX_CELEBA]
    if not candidatos:
        return ""
    return "%06d" % int(max(candidatos, key=len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=PASTA_DATASET)
    ap.add_argument("--saida", default=ARQUIVO_SAIDA)
    a = ap.parse_args()

    linhas, contagem, sem_indice = [], Counter(), 0
    for sub in SUBCONJUNTOS:
        for classe in CLASSES:
            pasta = os.path.join(a.dataset, sub, classe)
            if not os.path.isdir(pasta):
                raise FileNotFoundError(f"Pasta nao encontrada: {pasta}")
            for nome in sorted(os.listdir(pasta)):
                if not nome.lower().endswith(EXTENSOES) or not os.path.isfile(os.path.join(pasta, nome)):
                    continue
                idx = indice_celeba(nome)
                sem_indice += idx == ""
                linhas.append([sub, classe, nome, f"{sub}/{classe}/{nome}", idx])
                contagem[(sub, classe)] += 1

    # Conferencias: cada fotografia aparece no maximo uma vez por classe
    repetidos = Counter((l[1], l[4]) for l in linhas if l[4])
    repetidos = sum(1 for v in repetidos.values() if v > 1)

    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subconjunto", "classe", "arquivo", "caminho_relativo", "indice_celeba"])
        w.writerows(linhas)

    print("Imagens por subconjunto e classe (encontrado / artigo):")
    tudo_ok = True
    for chave, esperado in ESPERADO.items():
        n = contagem[chave]
        marca = "ok" if n == esperado else "DIFERENTE"
        tudo_ok &= n == esperado
        print(f"  {chave[0]:5s} {chave[1]:8s} {n:7d} / {esperado:7d}  {marca}")
    print(f"Total: {len(linhas)} imagens")
    print(f"Arquivos sem indice do CelebA no nome: {sem_indice}")
    print(f"Fotografias repetidas dentro da mesma classe: {repetidos}")
    print("Tudo confere com a Tabela 2." if tudo_ok and not sem_indice and not repetidos
          else "ATENCAO: ha diferencas, confira antes de publicar.")
    print(f"Gravado: {a.saida}")


if __name__ == "__main__":
    main()
