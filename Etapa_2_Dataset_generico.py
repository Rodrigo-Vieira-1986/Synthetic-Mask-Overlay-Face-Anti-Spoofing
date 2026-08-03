"""
Este script organiza um conjunto de imagens em um dataset com estrutura padronizada.

Objetivo:
---------
1. Acessar duas pastas de origem contendo imagens:
   - "attack"    → rostos falsos (ataques)
   - "bonafide"  → rostos reais (classe anteriormente chamada "real")

2. Embaralhar as imagens de cada classe para garantir aleatoriedade na divisão.

3. Dividir as imagens em três conjuntos:
   - 70% para treino (train)
   - 20% para validação (val)
   - 10% para teste (test)

4. Copiar os arquivos embaralhados para a estrutura de pastas destino:

   Dataset/
   ├── train/
   │   ├── attack/
   │   └── bonafide/
   ├── val/
   │   ├── attack/
   │   └── bonafide/
   └── test/
       ├── attack/
       └── bonafide/

5. Exibir o progresso da execução em porcentagem (%) durante a cópia dos arquivos.

Observações:
------------
- O script usa `shutil.copy2` para preservar os metadados dos arquivos.
- Nenhum arquivo é movido ou removido das pastas originais.
- As pastas destino são criadas automaticamente, caso não existam.
- O progresso é exibido em tempo real no terminal.
- Para mover os arquivos em vez de copiar, troque `shutil.copy2` por `shutil.move`.
"""

import os
import random
import shutil

# ================== CONFIGURE AQUI ==================
# Caminho da pasta com as fotos Attack (rostos falsos) de entrada
src_attack = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_ATTACK"
# Caminho da pasta com as fotos Bonafide (rostos reais) de entrada
src_bonafide = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_BONAFIDE"

# Caminho da pasta onde sera criado o Dataset com train/val/test
dst_base = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET"

# Proporções de divisão
train_pct = 0.7
val_pct = 0.2
test_pct = 0.1

# Função responsável por processar cada classe (attack ou bonafide)
def process_images(class_name, src_path):
    # Lista todos os arquivos da pasta
    images = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    
    # Embaralha os arquivos
    random.shuffle(images)

    # Calcula os limites de divisão
    total = len(images)
    train_end = int(total * train_pct)
    val_end = train_end + int(total * val_pct)

    # Separa os conjuntos
    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    copied = 0  # contador de arquivos copiados

    # Copia os arquivos para os diretórios de destino
    for split, files in split_data.items():
        dst_folder = os.path.join(dst_base, split, class_name)
        os.makedirs(dst_folder, exist_ok=True)

        for file in files:
            shutil.copy2(
                os.path.join(src_path, file),
                os.path.join(dst_folder, file)
            )

            # Atualiza progresso
            copied += 1
            progress = (copied / total) * 100
            print(f"{class_name} -> {progress:.2f}% concluído", end='\r')

    # Resumo final da classe
    print(f"\n{class_name.upper()} - Total: {total}, Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")


# Execução para ambas as classes
process_images('attack', src_attack)
process_images('bonafide', src_bonafide)
