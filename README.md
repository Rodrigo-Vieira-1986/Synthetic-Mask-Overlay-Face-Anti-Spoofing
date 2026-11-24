# Ataques de Apresentação por Máscara Facial Sintética a partir de Identidades Reais: um Novo Dataset e Benchmark Preliminar com CNN/Transformer

*Presentation Attacks via Synthetic Facial Masks from Real Identities: A New Dataset and Preliminary CNN/Transformer Benchmark* (VISAPP 2027)

> **Aviso:** As orientações relativas as imagens completas do dataset, arquivos de anotações e o data card estão hospedados e disponíveis no Kaggle em: [Kaggle Dataset](https://www.kaggle.com/datasets/rodrigovieira1986/synthetic-mask-overlay-face-anti-spoofing).

* **Última atualização:** 2026-11-24
* **Autor(es):** Rodrigo Vieira dos Santos, Erikson Freitas de Morais e Simone Bello Kaminski Aires (DAINF, UTFPR, Campus Ponta Grossa)
* **Licença:** CC BY-NC 4.0

---

## Descrição

Dataset para pesquisa em detecção de ataques de apresentação facial (Face Anti-Spoofing / Presentation Attack Detection - PAD), com três pastas de origem:

* **Bonafide** (`label=bonafide`): fotos de rosto reais.
* **Attack** (`label=attack`): a mesma foto original com o rosto (região do oval facial, sem os olhos) extraído e sobreposto de volta sobre ela, simulando digitalmente um ataque de apresentação.
* **Mask** (label vazio, `is_classification_sample=False`): o recorte do rosto (PNG com canal alfa) usado para gerar o Attack. Não é uma amostra de classificação — é o material que documenta a metodologia de geração do ataque, incluído para transparência e reprodutibilidade.

### Metodologia de geração dos ataques (importante para quem for usar o dataset)

As imagens da classe `attack` não são fotos de ataques físicos reais (não são impressões, reproduções em tela/replay, nem máscaras físicas fotografadas). Elas foram geradas sinteticamente a partir das próprias fotos `bonafide`: primeiro o rosto é segmentado com o MediaPipe Face Mesh (landmarks do oval do rosto, excluindo a região dos olhos), depois esse recorte passa por um leve ajuste de temperatura de cor (no máximo 2,1% por canal) e por contraste (fator 1,2) e brilho (+20), e por fim é ampliado em 1,15 e sobreposto de volta na foto original por *alpha blending*. O realce de textura e a suavização existem no código, mas não são acionados na configuração usada (`COLOR_TEMP = 7000`, `SMOOTHNESS = 0`). Isso deve ficar claro para quem for usar este dataset para treinar ou comparar detectores de PAD, já que o domínio (ataque sintético/digital) é diferente de datasets com ataques físicos capturados por câmera.

As anotações completas de cada imagem estão no arquivo `annotations.csv`, dentro de `dados_csv.zip`.

---

## Arquivos deste repositório

| Arquivo | Conteúdo |
| :--- | :--- |
| `Etapa_1_Gerador_Rostos_generico.py` | Gera as imagens Attack e Mask a partir das fotos do CelebA. |
| `Etapa_2_Dataset_generico.py` | Monta a divisão treino/validação/teste (70/20/10, aleatória por classe). |
| `Etapa_3_Treinamento_Redes_CNN_generico.py` | Treino das CNNs (MobileNetV2, ResNet50, Xception). |
| `Etapa_4_Inferencia_CNNs_V2_generico.py` | Avaliação das CNNs no teste. |
| `Etapa_5_Treinamento_Transformers_v2_generico.py` | Treino dos Transformers (DeiT-Small, Swin-Tiny, ViT-B/16). |
| `Etapa_6_Inferencia_Transformers_V2_generico.py` | Avaliação dos Transformers no teste. |
| `Etapa_7_Exportar_Particao_generico.py` | Grava a divisão usada nos experimentos em `particao_experimentos.csv`. |
| `Etapa_8a_Validacao_CNNs_generico.py` | Gera as predições das CNNs na validação. No artigo foi usada `--amostra 5000` (5.000 imagens por classe, semente 42). |
| `Etapa_8b_Validacao_Transformers_generico.py` | O mesmo para os Transformers. |
| `Etapa_9_Calibrar_Limiar_EER_generico.py` | Calibra o limiar do EER na validação e aplica no teste (coluna HTER (val.) da Tabela 5). |
| `Etapa_8_comum_validacao.py` | Funções usadas pelas Etapas 8a e 8b (não é executado sozinho). |
| `medir_vazamento.py` | Reproduz a análise de vazamento da Tabela 3 do artigo. |
| `dados_csv.zip` | `annotations.csv` (metadados de cada imagem) e `particao_experimentos.csv` (divisão usada nos experimentos). |

---

## ⚠️ Sobre as imagens não estarem incluídas neste repositório

As fotos `bonafide` de origem vêm do dataset CelebA (Liu et al., 2015), na versão alinhada e recortada (`img_align_celeba`). Os termos de uso do CelebA restringem o uso a pesquisa não-comercial e proíbem a redistribuição de suas imagens e de qualquer dado derivado delas (o que inclui as imagens `Attack` e `Mask` deste dataset, já que são geradas a partir das fotos do CelebA). Por isso, as imagens em si não são redistribuídas aqui — apenas o arquivo `annotations.csv` (metadados) e os scripts de geração.

**Para reproduzir o dataset completo com as imagens:**

1. Baixe a versão alinhada e recortada (`img_align_celeba`), disponibilizada pelo próprio autor do CelebA no [Link de Download CelebA](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ).
2. Rode o script de geração (`Etapa_1_Gerador_Rostos_generico.py`) sobre essas fotos, apontando para a pasta onde você salvou o CelebA.
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

## Divisão treino/validação/teste usada no artigo

O script `Etapa_2_Dataset_generico.py` não fixa semente aleatória. Por isso, a divisão exata usada nos experimentos do artigo está publicada em `particao_experimentos.csv` (dentro de `dados_csv.zip`), com as colunas:

| Coluna | Descrição |
| :--- | :--- |
| `subconjunto` | `train`, `val` ou `test`. |
| `classe` | `attack` ou `bonafide`. |
| `arquivo` | Nome do arquivo da imagem. |
| `caminho_relativo` | Caminho a partir da pasta `Dataset`. |
| `indice_celeba` | Número da foto de origem no CelebA, com 6 dígitos (permite cruzar com `identity_CelebA.txt`). |

| Subconjunto | Bonafide | Attack | Total |
| :--- | :--- | :--- | :--- |
| train | 141.819 | 141.444 | 283.263 |
| val | 40.519 | 40.412 | 80.931 |
| test | 20.261 | 20.207 | 40.468 |

**Atenção ao vazamento.** A divisão é aleatória por classe e não separa identidades nem as duas versões de cada foto. 99,94% das identidades do teste também aparecem no treino (9.300 de 9.306), e 69,74% das imagens de teste têm a outra versão da mesma foto no treino (28.222 de 40.468). Os resultados do artigo são, portanto, um limite superior num cenário com sujeitos já vistos. Para novos experimentos, recomenda-se uma divisão por sujeito.

Para reproduzir esses números (o arquivo `identity_CelebA.txt` é obtido no site do CelebA):

```
python medir_vazamento.py --particao particao_experimentos.csv --identidades identity_CelebA.txt
```

---

## Ética e consentimento

As fotos `bonafide` são provenientes do CelebA, um dataset de rostos coletados da internet (imagens de figuras públicas). Não há documentação de consentimento individual das pessoas fotografadas para o uso original nem para usos derivados como este — essa é uma limitação conhecida e discutida publicamente sobre o CelebA, não algo específico deste trabalho. Por isso as imagens não são redistribuídas; apenas metadados e código são publicados.

Este dataset foi produzido no âmbito das atividades de Mestrado do Programa de Pós-Graduação em Ciência da Computação (PPGCC-PG).

O `annotations.csv` publicado não contém nenhum dado pessoal além do que é extraído diretamente dos arquivos de imagem (caminho, dimensões, formato, tamanho) — não há nomes, e-mails, localização ou qualquer outro identificador pessoal.

---

## Como citar

O artigo foi aceito no VISAPP 2027 (22nd International Conference on Computer Vision Theory and Applications, Valletta, Malta, 26 a 28 de fevereiro de 2027). A referência completa, com páginas e DOI, será adicionada após a publicação. Por enquanto, use:

> SANTOS, Rodrigo Vieira dos; MORAIS, Erikson Freitas de; AIRES, Simone Bello Kaminski. Presentation Attacks via Synthetic Facial Masks from Real Identities: A New Dataset and Preliminary CNN/Transformer Benchmark. In: *Proceedings of the 22nd International Conference on Computer Vision Theory and Applications (VISAPP 2027)*. SCITEPRESS, 2027. No prelo.

E cite também o dataset de origem das imagens:

> LIU, Ziwei; LUO, Ping; WANG, Xiaogang; TANG, Xiaoou. Deep Learning Face Attributes in the Wild. *Proceedings of International Conference on Computer Vision (ICCV)*, 2015.

---

## Limitações conhecidas

* **Ataques sintéticos, não físicos:** a classe `attack` é gerada digitalmente por sobreposição, não fotografada de um ataque de apresentação real. Modelos treinados aqui podem não generalizar para ataques físicos reais (impressão, replay, máscara física) sem validação adicional.
* **`source_id_hint` não é confiável para rastrear a foto bonafide de origem:** no pipeline original, os arquivos de Attack/Mask usam os dígitos do nome do arquivo bonafide antes da renomeação sequencial, então o número pode não bater com o `bonafide_XXXXX.jpg` final.
* **Condições de captura não controladas:** as fotos `bonafide` vêm do CelebA, um dataset de imagens coletadas da internet, não capturadas pelos autores deste trabalho. Por isso não há controle nem registro de iluminação, dispositivo de captura ou distância câmera-rosto por imagem — essas condições variam de foto para foto, herdadas do dataset de origem.
* **Desbalanceamento demográfico herdado do CelebA:** a distribuição demográfica dos indivíduos segue a do CelebA, que é documentada na literatura como predominantemente composta por peles mais claras (ver GEORGOPOULOS, M. et al. *Mitigating Demographic Bias in Facial Datasets with Style-Based Multi-attribute Transfer*. International Journal of Computer Vision, v. 129, n. 7, 2021). Essa distribuição não foi ajustada neste trabalho e pode afetar a generalização do modelo para outras populações.
