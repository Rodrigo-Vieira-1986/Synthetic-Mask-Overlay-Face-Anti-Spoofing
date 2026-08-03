# Ataques de Apresentação por Máscara Facial Sintética a partir de Identidades Reais: um Novo Dataset e Benchmark Preliminar com CNN/Transformer

> **Aviso:** As orientações relativas as imagens completas do dataset, arquivos de anotações e o data card estão hospedados e disponíveis no Kaggle em: [Kaggle Dataset](https://www.kaggle.com/datasets/rodrigovieira1986/synthetic-mask-overlay-face-anti-spoofing).

* **Última atualização:** 2026-08-02
* **Autor(es):** Rodrigo Vieira dos Santos
* **Licença:** CC BY-NC 4.0

---

## Descrição

Dataset para pesquisa em detecção de ataques de apresentação facial (Face Anti-Spoofing / Presentation Attack Detection - PAD), com três pastas de origem:

* **Bonafide** (`label=bonafide`): fotos de rosto reais.
* **Attack** (`label=attack`): a mesma foto original com o rosto (região do oval facial, sem os olhos) extraído e sobreposto de volta sobre ela, simulando digitalmente um ataque de apresentação.
* **Mask** (label vazio, `is_classification_sample=False`): o recorte do rosto (PNG com canal alfa) usado para gerar o Attack. Não é uma amostra de classificação — é o material que documenta a metodologia de geração do ataque, incluído para transparência e reprodutibilidade.

### Metodologia de geração dos ataques (importante para quem for usar o dataset)

As imagens da classe `attack` não são fotos de ataques físicos reais (não são impressões, reproduções em tela/replay, nem máscaras físicas fotografadas). Elas foram geradas sinteticamente a partir das próprias fotos `bonafide`: primeiro o rosto é segmentado com o MediaPipe Face Mesh (landmarks do oval do rosto, excluindo a região dos olhos), depois esse recorte passa por ajustes de temperatura de cor, brilho/contraste e realce de textura (para simular material artificial), e por fim é sobreposto de volta na foto original. Isso deve ficar claro para quem for usar este dataset para treinar ou comparar detectores de PAD, já que o domínio (ataque sintético/digital) é diferente de datasets com ataques físicos capturados por câmera.

As anotações completas de cada imagem estão no arquivo `annotations.csv`.

---

## ⚠️ Sobre as imagens não estarem incluídas neste repositório

As fotos `bonafide` de origem vêm do dataset CelebA (Liu et al., 2015), na versão alinhada e recortada (`img_align_celeba`). Os termos de uso do CelebA restringem o uso a pesquisa não-comercial e proíbem a redistribuição de suas imagens e de qualquer dado derivado delas (o que inclui as imagens `Attack` e `Mask` deste dataset, já que são geradas a partir das fotos do CelebA). Por isso, as imagens em si não são redistribuídas aqui — apenas o arquivo `annotations.csv` (metadados) e os scripts de geração.

**Para reproduzir o dataset completo com as imagens:**

1. Baixe a versão alinhada e recortada (`img_align_celeba`), disponibilizada pelo próprio autor do CelebA no [Link de Download CelebA](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ).
2. Rode o script de geração (`Etapa_1_Gerador_Rostos.py`) sobre essas fotos, apontando para a pasta onde você salvou o CelebA.
3. O `annotations.csv` publicado aqui pode então ser usado para conferir se sua reprodução bate com o dataset original (mesmos nomes de arquivo, mesma contagem por classe).

Para mais detalhes, acesse a [Página oficial do projeto CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

---

## Estatísticas do dataset

* **Total de imagens:** 606.725
* **Imagens não legíveis/corrompidas:** 0

### Distribuição por pasta de origem

| Pasta | Quantidade |
| :--- | :--- |
| **Attack** | 202.063 |
| **Bonafide** | 202.599 |
| **Mask** | 202.063 |

### Distribuição por rótulo (label)

| Label | Quantidade |
| :--- | :--- |
| **attack** | 202.063 |
| **bonafide** | 202.599 |
| **n/a (mask asset)** | 202.063 |

---

## Estrutura do CSV (annotations.csv)

| Coluna | Descrição |
| :--- | :--- |
| `image_id` | Identificador único da imagem dentro do dataset. |
| `relative_path` | Caminho relativo à raiz do dataset (portável entre sistemas). |
| `filename` | Nome do arquivo de imagem. |
| `class_folder` | Pasta de origem: Bonafide, Attack ou Mask. |
| `label` | Rótulo binário: bonafide ou attack. Vazio para Mask (não é amostra de classificação). |
| `is_classification_sample` | True para Bonafide/Attack, False para Mask. |
| `source_id_hint` | Dígitos extraídos do nome do arquivo. Não usar como garantia de correspondência exata com a foto bonafide de origem (ver Limitações). |
| `width`, `height` | Dimensões da imagem em pixels. |
| `channels` | Modo de cor da imagem (ex.: RGB, RGBA). |
| `file_format` | Formato do arquivo (ex.: JPEG, PNG). |
| `file_size_bytes` | Tamanho do arquivo em bytes. |

---

## Divisão treino/validação/teste

Este dataset é publicado sem uma divisão obrigatória de treino/validação/teste — cada pesquisador pode fazer a própria divisão (ou usar validação cruzada) de acordo com o experimento.

---

## Ética e consentimento

As fotos `bonafide` são provenientes do CelebA, um dataset de rostos coletados da internet (imagens de figuras públicas). Não há documentação de consentimento individual das pessoas fotografadas para o uso original nem para usos derivados como este — essa é uma limitação conhecida e discutida publicamente sobre o CelebA, não algo específico deste trabalho. Por isso as imagens não são redistribuídas; apenas metadados e código são publicados.

Este dataset foi produzido no âmbito das atividades de Mestrado do Programa de Pós-Graduação em Ciência da Computação (PPGCC-PG).

O `annotations.csv` publicado não contém nenhum dado pessoal além do que é extraído diretamente dos arquivos de imagem (caminho, dimensões, formato, tamanho) — não há nomes, e-mails, localização ou qualquer outro identificador pessoal.

---

## Como citar

Este trabalho ainda está em andamento e não possui artigo/dissertação formalmente publicado até o momento. A citação definitiva será adicionada aqui assim que a publicação estiver disponível. Por enquanto, se for referenciar este dataset, use:

> Rodrigo Vieira dos Santos (2026). *Ataques de Apresentação por Máscara Facial Sintética a partir de Identidades Reais: um Novo Dataset e Benchmark Preliminar com CNN/Transformer*. Dataset e código-fonte (não publicado).

E cite também o dataset de origem das imagens:

> LIU, Ziwei; LUO, Ping; WANG, Xiaogang; TANG, Xiaoou. Deep Learning Face Attributes in the Wild. *Proceedings of International Conference on Computer Vision (ICCV)*, 2015.

---

## Limitações conhecidas

* **Ataques sintéticos, não físicos:** a classe `attack` é gerada digitalmente por sobreposição, não fotografada de um ataque de apresentação real. Modelos treinados aqui podem não generalizar para ataques físicos reais (impressão, replay, máscara física) sem validação adicional.
* **`source_id_hint` não é confiável para rastrear a foto bonafide de origem:** no pipeline original, os arquivos de Attack/Mask usam os dígitos do nome do arquivo bonafide antes da renomeação sequencial, então o número pode não bater com o `bonafide_XXXXX.jpg` final.
* **Condições de captura não controladas:** as fotos `bonafide` vêm do CelebA, um dataset de imagens coletadas da internet, não capturadas pelos autores deste trabalho. Por isso não há controle nem registro de iluminação, dispositivo de captura ou distância câmera-rosto por imagem — essas condições variam de foto para foto, herdadas do dataset de origem.
* **Desbalanceamento demográfico herdado do CelebA:** a distribuição demográfica dos indivíduos segue a do CelebA, que é documentada na literatura como predominantemente composta por peles mais claras (ver GEORGOPOULOS, M. et al. *Mitigating Demographic Bias in Facial Datasets with Style-Based Multi-attribute Transfer*. International Journal of Computer Vision, v. 129, n. 7, 2021). Essa distribuição não foi ajustada neste trabalho e pode afetar a generalização do modelo para outras populações.
