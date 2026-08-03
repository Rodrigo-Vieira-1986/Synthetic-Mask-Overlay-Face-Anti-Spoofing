"""
Este código realiza o processamento de imagens faciais para geração de um dataset sintético
voltado à detecção de spoofing facial (classificação entre rostos reais e falsos).

Etapas do processamento:

1. Leitura das imagens da pasta Bonafide (imagens reais).
2. Detecção facial utilizando MediaPipe Face Mesh.
3. Extração de landmarks faciais.
4. Construção de uma máscara baseada no contorno do rosto (face oval),
   removendo as regiões dos olhos.
5. Aplicação de efeitos visuais para simular materiais artificiais, incluindo:
   - ajuste de temperatura de cor
   - realce de textura (efeito borracha/plástico)
   - ajuste de brilho e contraste
6. Geração de uma imagem com canal alfa representando a máscara facial.
7. Sobreposição da máscara sobre a imagem original, simulando um ataque de spoofing.
8. Salvamento das saídas:
   - Mask (máscara facial)
   - Attack (imagem com máscara sobreposta)

Ao final do processamento:
9. Os arquivos da pasta Bonafide são renomeados automaticamente para o padrão:
   bonafide_00001.jpg, bonafide_00002.jpg, ...

Observações:
- A resolução original das imagens é preservada.
- O redimensionamento deve ser feito apenas no treinamento.
- O pipeline mantém características importantes como bordas artificiais e inconsistências visuais.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

# ========= CONFIGURAÇÕES ========= #
# ================== CONFIGURE AQUI ==================
# Caminho da pasta com as fotos Bonafide (rostos reais) de entrada
INPUT_FOLDER = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_BONAFIDE"
# Caminho da pasta onde serao salvas as mascaras geradas (Mask)
OUTPUT_FOLDER = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_MASK"
# Caminho da pasta onde serao salvas as imagens Attack geradas
OUTPUT_FOLDER_2 = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_ATTACK"

# ========= PARÂMETROS AJUSTÁVEIS ========= #
MASK_SCALE = 1.15
OUTPUT_SCALE = 1.15
OVERLAY_SCALE = 1.15

COLOR_TEMP = 7000
COLOR_INTENSITY = 1.0
BRIGHTNESS = 20
CONTRAST = 1.2
SMOOTHNESS = 0

# ========= INICIALIZAÇÃO ========= #
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_2, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_EYE = list(set([idx for pair in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for idx in pair]))
RIGHT_EYE = list(set([idx for pair in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for idx in pair]))

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ========= FUNÇÕES AUXILIARES ========= #
def aplicar_efeito_plastico(imagem, brilho=0, contraste=1.0, suavidade=100):
    img = cv2.convertScaleAbs(imagem, alpha=contraste, beta=brilho)

    if suavidade > 0:
        img = cv2.GaussianBlur(img, (0, 0), suavidade)
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    return img

def ajustar_temperatura_cor(imagem, temp_k, intensidade):
    if temp_k == 6500 and intensidade == 1.0:
        return imagem.copy()

    temp_k = np.clip(temp_k, 2500, 15000)
    intensidade = np.clip(intensidade, 0.1, 2.0)

    if temp_k < 6500:
        r = 1.0
        g = 0.39 * np.log(temp_k/100) - 0.29
        b = 0.543 * np.log(temp_k/100) - 0.41
    else:
        ratio = (temp_k - 6500) / 3500
        r = 1.0 - (0.15 * ratio)
        g = 1.0 - (0.05 * ratio)
        b = 1.0 + (0.05 * ratio)

    img = imagem.astype(np.float32)

    for canal, valor in zip([2, 1, 0], [r, g, b]):
        img[:, :, canal] = np.clip(img[:, :, canal] * valor * intensidade, 0, 255)

    return img.astype(np.uint8)

def realcar_borracha(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.detailEnhance(img, sigma_s=5, sigma_r=0.12)

def sobrepor_imagens(original_path, mascara_path, output_path):
    original = cv2.imread(original_path, cv2.IMREAD_COLOR)
    mascara = cv2.imread(mascara_path, cv2.IMREAD_UNCHANGED)

    if original is None or mascara is None:
        print("Erro ao carregar imagens para sobreposição")
        return

    h, w = original.shape[:2]

    mascara = cv2.resize(mascara, (int(w * OVERLAY_SCALE), int(h * OVERLAY_SCALE)))

    offset_x = int((w * OVERLAY_SCALE - w) / 2)
    offset_y = int((h * OVERLAY_SCALE - h) / 2)

    temp = np.zeros((mascara.shape[0], mascara.shape[1], 4), dtype=np.uint8)

    temp[offset_y:offset_y+h, offset_x:offset_x+w, :3] = original
    temp[offset_y:offset_y+h, offset_x:offset_x+w, 3] = 255

    alpha = mascara[:, :, 3] / 255.0

    for c in range(3):
        temp[:, :, c] = (1 - alpha) * temp[:, :, c] + alpha * mascara[:, :, c]

    resultado = temp[offset_y:offset_y+h, offset_x:offset_x+w]

    cv2.imwrite(output_path, resultado)

# ========= FUNÇÃO PRINCIPAL ========= #
def criar_mascara_facial():

    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(arquivos)
    contador = 0

    for arquivo in arquivos:
        contador += 1
        progresso = (contador / total) * 100
        print(f"Processando {contador}/{total} ({progresso:.2f}%) - {arquivo}")

        try:
            img_path = os.path.join(INPUT_FOLDER, arquivo)
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Erro ao ler: {arquivo}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            resultados = face_mesh.process(img)

            if not resultados.multi_face_landmarks:
                continue

            landmarks = resultados.multi_face_landmarks[0]
            mascara = np.zeros((h, w), dtype=np.uint8)

            pontos = np.array([
                (int(landmarks.landmark[i].x * w),
                 int(landmarks.landmark[i].y * h))
                for i in FACE_OVAL
            ], dtype=np.int32)

            cv2.fillConvexPoly(mascara, pontos, 255)

            for olho in [LEFT_EYE, RIGHT_EYE]:
                pontos_olho = np.array([
                    (int(landmarks.landmark[i].x * w),
                     int(landmarks.landmark[i].y * h))
                    for i in olho
                ], dtype=np.int32)

                cv2.fillConvexPoly(mascara, pontos_olho, 0)

            resultado = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
            resultado[:, :, 3] = mascara

            if COLOR_TEMP != 6500 or COLOR_INTENSITY != 1.0:
                rgb = cv2.cvtColor(resultado, cv2.COLOR_BGRA2RGB)
                rgb = ajustar_temperatura_cor(rgb, COLOR_TEMP, COLOR_INTENSITY)

                if COLOR_TEMP > 7000:
                    rgb = realcar_borracha(rgb)

                rgb = aplicar_efeito_plastico(rgb, BRIGHTNESS, CONTRAST, SMOOTHNESS)

                resultado = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
                resultado[:, :, 3] = mascara

            if OUTPUT_SCALE != 1.0:
                resultado = cv2.resize(
                    resultado,
                    (int(w * OUTPUT_SCALE), int(h * OUTPUT_SCALE)),
                    interpolation=cv2.INTER_LANCZOS4
                )

            nome_base = ''.join(filter(str.isdigit, os.path.splitext(arquivo)[0]))

            nome_saida = f"mask_{nome_base}.png"
            saida_path = os.path.join(OUTPUT_FOLDER, nome_saida)

            cv2.imwrite(saida_path, resultado)

            nome_saida_2 = f"attack_{nome_base}.jpg"
            saida_path_2 = os.path.join(OUTPUT_FOLDER_2, nome_saida_2)

            sobrepor_imagens(img_path, saida_path, saida_path_2)

        except Exception as e:
            print(f"Erro: {str(e)}")

# ========= RENOMEAR BONAFIDE ========= #
def renomear_bonafide():
    arquivos = sorted(os.listdir(INPUT_FOLDER))

    for i, arquivo in enumerate(arquivos):
        caminho_antigo = os.path.join(INPUT_FOLDER, arquivo)
        novo_nome = f"bonafide_{i+1:05d}.jpg"
        caminho_novo = os.path.join(INPUT_FOLDER, novo_nome)

        if caminho_antigo != caminho_novo:
            os.rename(caminho_antigo, caminho_novo)

if __name__ == "__main__":
    criar_mascara_facial()
    renomear_bonafide()
    face_mesh.close()
    print("Processamento concluído")
