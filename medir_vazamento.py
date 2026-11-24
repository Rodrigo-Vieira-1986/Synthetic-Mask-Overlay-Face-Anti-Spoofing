# -*- coding: utf-8 -*-
"""
Mede o vazamento de identidade e de par de origem na particao usada no artigo
==============================================================================

Reproduz a Tabela 3 do artigo (VISAPP 2027) a partir de dois arquivos:

  particao_experimentos.csv  particao treino/validacao/teste usada nos
                             experimentos (publicada junto com este codigo)
  identity_CelebA.txt        anotacao oficial de identidades do CelebA
                             (baixe do site do CelebA; nao e redistribuida aqui)

Nao precisa das imagens. So le os dois arquivos de texto.

Uso:
    python medir_vazamento.py --particao particao_experimentos.csv --identidades identity_CelebA.txt

Sem --identidades, mede apenas o vazamento de par de origem (a outra versao da
mesma foto no treino), que nao depende das identidades.
"""
import argparse
import csv
from collections import defaultdict


def carregar_particao(caminho):
    """Retorna {subconjunto: {classe: set(indice_celeba)}}."""
    part = defaultdict(lambda: defaultdict(set))
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            part[linha["subconjunto"]][linha["classe"]].add(linha["indice_celeba"])
    return part


def carregar_identidades(caminho):
    """Retorna {indice_celeba com 6 digitos: id da pessoa}."""
    ident = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            partes = linha.split()
            if len(partes) == 2:
                ident["%06d" % int(partes[0].split(".")[0])] = int(partes[1])
    return ident


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particao", default="particao_experimentos.csv")
    ap.add_argument("--identidades", default=None, help="identity_CelebA.txt (opcional)")
    a = ap.parse_args()

    part = carregar_particao(a.particao)
    print("Imagens por subconjunto e classe:")
    for sub in ("train", "val", "test"):
        print("  %-5s attack=%7d  bonafide=%7d" % (sub, len(part[sub]["attack"]), len(part[sub]["bonafide"])))

    treino = part["train"]
    teste = part["test"]

    # Vazamento de par de origem: a outra versao da mesma foto esta no treino
    print("\nVazamento de par de origem (teste):")
    total_vaz = total_img = 0
    for classe, outra in (("attack", "bonafide"), ("bonafide", "attack")):
        n = len(teste[classe])
        v = len(teste[classe] & treino[outra])
        total_vaz += v
        total_img += n
        print("  %-8s %6d de %6d (%.2f%%)" % (classe, v, n, pct(v, n)))
    print("  total    %6d de %6d (%.2f%%)" % (total_vaz, total_img, pct(total_vaz, total_img)))
    ambas = len(teste["attack"] & teste["bonafide"])
    print("  fotos com as duas versoes no proprio teste: %d" % ambas)

    if not a.identidades:
        print("\nSem --identidades: vazamento de identidade nao calculado.")
        return

    ident = carregar_identidades(a.identidades)

    def pessoas(sub):
        s = set()
        for classe in ("attack", "bonafide"):
            s |= {ident[i] for i in part[sub][classe] if i in ident}
        return s

    p_treino = pessoas("train")
    p_teste = pessoas("test")
    comuns = p_teste & p_treino
    print("\nVazamento de identidade:")
    print("  identidades no treino: %d" % len(p_treino))
    print("  identidades no teste:  %d" % len(p_teste))
    print("  identidades do teste tambem no treino: %d (%.2f%%)" % (len(comuns), pct(len(comuns), len(p_teste))))
    print("  identidades exclusivas do teste: %d" % len(p_teste - p_treino))


if __name__ == "__main__":
    main()
