"""
TREINAMENTO_TRANSFORMERS_V2.PY
Versao robusta para dissertacao

Modelos:
1 - ViT-B16
2 - DeiT-Small
3 - Swin-Tiny

Recursos:
- Mac MPS / CUDA / CPU automatico
- Resume automatico por fase
- EarlyStopping
- CSV continuo
- Fine-tuning
- Accuracy / Precision / Recall / AUC
- Salvamento .pth
- Arquivo info.txt
"""

# =====================================================
# IMPORTS
# =====================================================

import os
import json
import sys
import platform
import time
import random
import numpy as np
import pandas as pd
import torch
import timm

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import torchvision
from torchvision import datasets
from torchvision import transforms

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# =====================================================
# REPRODUTIBILIDADE
# =====================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# =====================================================
# CONFIGURAÇÕES GLOBAIS
# =====================================================

IMG_SIZE = 224
EPOCHS = 1000
FINE_TUNE_EPOCHS = 20
FT_PATIENCE = 3
PATIENCE = 10
FINE_TUNE_BLOCKS = {
    "vit": 2,
    "deit": 2,
    "swin": 1
}
NUM_CLASSES = 2
POSITIVE_CLASS_NAME = "attack"
CSV_COLUNAS = [
    "epoch",
    "accuracy",
    "auc",
    "f1_score",
    "loss",
    "lr",
    "scheduler_step",
    "precision",
    "recall",
    "val_accuracy",
    "val_auc",
    "val_f1_score",
    "val_loss",
    "val_precision",
    "val_recall",
    "trainable_params",
]

DEBUG_TIMING = False

# =====================================================
# FUNCOES AUXILIARES
# =====================================================

def carregar_historico(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).to_dict("records")
    return []


def salvar_historico(csv_path, historico):
    df = pd.DataFrame(historico)

    for coluna in CSV_COLUNAS:
        if coluna not in df.columns:
            df[coluna] = np.nan

    df = df[CSV_COLUNAS]
    df.to_csv(csv_path, index=False)


def ultimo_epoch_csv(csv_path):
    if not os.path.exists(csv_path):
        return 0

    try:
        df = pd.read_csv(csv_path)
        if df.empty or "epoch" not in df.columns:
            return 0

        epoch_val = df["epoch"].max()
        if pd.isna(epoch_val):
            return 0
        return int(epoch_val)
    except Exception:
        return 0


def carregar_estado(path, model, optimizer=None, device=None, scheduler=None):
    if not os.path.exists(path):
        return None

    if device is None:
        device = torch.device("cpu")

    try:
        state = torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except Exception as e:
        print(f"  Erro ao carregar {os.path.basename(path)}: {e}. Arquivo ignorado.")
        return None

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None:
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        else:
            print("  Scheduler state nao encontrado no checkpoint. Scheduler sera reiniciado.")

    return state


def salvar_estado(path, model, optimizer, epoch, best_loss, patience_count, phase, model_name="", best_epoch=0, scheduler_state_dict=None):
    # epoch: ultima epoca concluida (1-indexed).
    save_dict = {
        "phase": phase,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "num_classes": NUM_CLASSES,
        "input_size": IMG_SIZE
    }
    if scheduler_state_dict is not None:
        save_dict["scheduler_state_dict"] = scheduler_state_dict
    torch.save(save_dict, path)


def congelar_base(model):
    for p in model.parameters():
        p.requires_grad = False

    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True


def liberar_fine_tuning(model, num_blocks=2):
    for p in model.parameters():
        p.requires_grad = False

    tem_blocks = hasattr(model, "blocks")
    tem_layers = hasattr(model, "layers")

    if tem_blocks:
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            for p in classifier.parameters():
                p.requires_grad = True

        for bloco in model.blocks[-num_blocks:]:
            for p in bloco.parameters():
                p.requires_grad = True

    elif tem_layers:
        head = model.head
        if isinstance(head, nn.Module):
            for p in head.parameters():
                p.requires_grad = True

        for layer in model.layers[-num_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True


def criar_otimizador(model, lr):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

# =====================================================
# FUNÇÃO AUXILIAR - FORMATAÇÃO DE TEMPO
# =====================================================

def formatar_tempo(segundos):
    segundos = max(0, int(round(segundos)))
    horas, resto = divmod(segundos, 3600)
    minutos, seg = divmod(resto, 60)

    if horas > 0:
        return f"{horas:02d}:{minutos:02d}:{seg:02d}"
    return f"{minutos:02d}:{seg:02d}"

# =====================================================
# FUNÇÃO AUXILIAR - CONVERSÃO NUMPY PARA JSON
# =====================================================

def converter_para_json(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: converter_para_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [converter_para_json(v) for v in obj]
    return obj


# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():

    # =====================================================
    # SEÇÃO 1: CONFIGURAÇÃO
    # =====================================================

    print("Escolha o modelo:")
    print("1 - ViT-B16")
    print("2 - DeiT-Small")
    print("3 - Swin-Tiny")

    opcao = input("Digite o numero do modelo: ")

    if opcao == "1":
        MODEL_NAME = "vit_base_patch16_224"
        PREFIX = "vit"
    elif opcao == "2":
        MODEL_NAME = "deit_small_patch16_224"
        PREFIX = "deit"
    elif opcao == "3":
        MODEL_NAME = "swin_tiny_patch4_window7_224"
        PREFIX = "swin"
    else:
        raise ValueError("Opcao invalida")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 8

    script_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(script_dir, PREFIX)
    os.makedirs(modelos_dir, exist_ok=True)

    print("1 - Mac / caminho portatil")
    print("2 - Windows / caminho fixo")

    amb = input("Ambiente: ")

    if amb == "1":
        base_dir = os.path.join(script_dir, "Dataset")
    elif amb == "2":
        # Caminho da pasta Dataset contendo train/val/test
        base_dir = r"COLOQUE_AQUI_O_CAMINHO_DA_PASTA_DATASET"
    else:
        raise ValueError("Ambiente invalido")

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    checkpoint_path = os.path.join(modelos_dir, f"{PREFIX}.pth")
    checkpoint_ft_path = os.path.join(modelos_dir, f"{PREFIX}_finetuning.pth")
    final_path = os.path.join(modelos_dir, f"{PREFIX}_final.pth")

    csv_fase1 = os.path.join(modelos_dir, f"{PREFIX}_fase1.csv")
    csv_ft = os.path.join(modelos_dir, f"{PREFIX}_finetuning.csv")

    backup_fase1_path = os.path.join(modelos_dir, f"{PREFIX}_backup_fase1.pth")
    backup_ft_path = os.path.join(modelos_dir, f"{PREFIX}_backup_finetuning.pth")

    info_file = os.path.join(modelos_dir, f"{PREFIX}_info.txt")

    # =====================================================
    # SEÇÃO 2: DATASET
    # =====================================================

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Diretório de treino não encontrado: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Diretório de validação não encontrado: {val_dir}")

    normalizacao_imagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=(0.6, 1.4), contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        normalizacao_imagenet
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalizacao_imagenet
    ])

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf)

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Classes de train e val nao coincidem")

    positive_idx = train_ds.class_to_idx.get(POSITIVE_CLASS_NAME, 1)

    print("Classes:", train_ds.class_to_idx)
    print("Classe positiva para metricas:", positive_idx)
    print("Amostras train:", len(train_ds))
    print("Amostras val:", len(val_ds))

    if DEVICE.type == "cuda":
        num_workers = max(2, os.cpu_count() // 2)
    else:
        num_workers = 2
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda"
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )

    print("Batch size:", BATCH_SIZE)
    print("Workers DataLoader:", num_workers)
    print("Batches por epoch (train):", len(train_loader))
    print("Batches por epoch (val):", len(val_loader))

    total_epochs_phase1 = max(0, EPOCHS)
    total_steps_phase1 = len(train_loader) * total_epochs_phase1
    print("Estimativa total fase 1 (steps):", f"{total_steps_phase1:,}")
    if len(train_loader) > 10000:
        print("Aviso: a fase 1 tem muitas iteracoes por epoch; no Mac isso pode levar bastante tempo.")

    # =====================================================
    # SEÇÃO 3: CONSTRUÇÃO DO MODELO
    # =====================================================

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    congelar_base(model)
    model = model.to(DEVICE)

    PARAMS = sum(p.numel() for p in model.parameters())
    TRAINABLE_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
    TRAINABLE_PARAMS_FASE1 = TRAINABLE_PARAMS

    print("Parametros:", f"{PARAMS:,}")
    print("Parametros treinaveis fase 1:", f"{TRAINABLE_PARAMS:,}")

    # =====================================================
    # BALANCEAMENTO DAS CLASSES
    # =====================================================

    class_counts = {}
    for cls_name, cls_idx in train_ds.class_to_idx.items():
        class_counts[cls_name] = np.sum(np.array(train_ds.targets) == cls_idx)

    print("\nDistribuição das classes (treino):")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} imagens")

    values = list(class_counts.values())
    if len(values) >= 2:
        ratio = max(values) / min(values)
        usar_class_weight = ratio > 1.5
    else:
        usar_class_weight = False

    if usar_class_weight:
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_ds.targets),
            y=train_ds.targets
        )
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        print("\nPesos calculados para cada classe:")
        for cls_name, cls_idx in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1]):
            print(f"  {cls_name} (índice {cls_idx}): {weights[cls_idx]:.4f}")
        print("\n-> class_weight será utilizado no treinamento")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nClasses balanceadas. class_weight não será utilizado.")
    print()

    optimizer = criar_otimizador(model, lr=1e-4)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type) if use_amp else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    def avaliar_loader(loader):
        model.eval()

        losses = []
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                out = model(x)
                loss = criterion(out, y)

                losses.append(loss.item())

                prob = torch.softmax(out, dim=1)
                pred = torch.argmax(prob, dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(prob[:, positive_idx].cpu().numpy())

        y_true_bin = [1 if y == positive_idx else 0 for y in y_true]
        y_pred_bin = [1 if y == positive_idx else 0 for y in y_pred]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        try:
            auc = roc_auc_score(y_true_bin, y_prob)
        except ValueError:
            auc = 0

        return np.mean(losses), acc, prec, rec, f1, auc

    def treinar_epoch(optimizer, scaler, use_amp):
        model.train()

        total_batches = len(train_loader)
        batch_inicio = time.time()

        losses_epoch = []
        y_true_epoch = []
        y_pred_epoch = []
        y_prob_epoch = []

        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(DEVICE.type):
                    out = model(imgs)
                    loss = criterion(out, labels)
            else:
                out = model(imgs)
                loss = criterion(out, labels)

            losses_epoch.append(loss.item())

            prob = torch.softmax(out.detach(), dim=1)
            pred = torch.argmax(prob, dim=1)

            y_true_epoch.extend(labels.cpu().numpy())
            y_pred_epoch.extend(pred.cpu().numpy())
            y_prob_epoch.extend(prob[:, positive_idx].cpu().numpy())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if batch_idx == 1 or batch_idx % 10 == 0:
                batch_elapsed = time.time() - batch_inicio
                eta_epoch = (batch_elapsed / batch_idx) * (total_batches - batch_idx) if batch_idx > 0 else 0
                print(
                    f"\r  batch {batch_idx}/{total_batches} | "
                    f"loss {loss.item():.4f} | "
                    f"eta epoca {formatar_tempo(eta_epoch)}",
                    end="", flush=True
                )

        print(
            f"\r  batch {total_batches}/{total_batches} | "
            f"loss --- | "
            f"eta epoca 00:00:00"
        )

        y_true_bin = [1 if y == positive_idx else 0 for y in y_true_epoch]
        y_pred_bin = [1 if y == positive_idx else 0 for y in y_pred_epoch]

        train_loss = np.mean(losses_epoch)
        train_acc = accuracy_score(y_true_epoch, y_pred_epoch)
        train_prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        train_rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        train_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        try:
            train_auc = roc_auc_score(y_true_bin, y_prob_epoch)
        except ValueError:
            train_auc = 0

        return train_loss, train_acc, train_prec, train_rec, train_f1, train_auc

    # =====================================================
    # SEÇÃO 4: FASE 1
    # =====================================================

    # =====================================================
    # COMPILAÇÃO (otimizador + loss)
    # =====================================================

    # optimizer e criterion já definidos acima

    # =====================================================
    # CONTINUAR DA ÚLTIMA ÉPOCA (FASE 1)
    # =====================================================

    best_loss = float("inf")
    best_epoch_fase1 = 0
    patience_count = 0
    historico = carregar_historico(csv_fase1)
    # start_epoch: ultima epoca concluida (1-indexed).
    start_epoch = ultimo_epoch_csv(csv_fase1)
    # proxima_epoca: primeira epoca a treinar (1-indexed). Padrao para fluxos sem backup.
    proxima_epoca = start_epoch + 1

    print("Verificando retomada da fase 1...", flush=True)
    estado_fase1 = carregar_estado(backup_fase1_path, model, optimizer, device=DEVICE, scheduler=scheduler)
    if estado_fase1 is not None:
        start_epoch = int(estado_fase1.get("epoch", start_epoch))
        best_loss = float(estado_fase1.get("best_loss", best_loss))
        best_epoch_fase1 = int(estado_fase1.get("best_epoch", best_epoch_fase1))
        patience_count = int(estado_fase1.get("patience_count", patience_count))

        epoch_csv_fase1 = ultimo_epoch_csv(csv_fase1)
        if epoch_csv_fase1 > 0 and epoch_csv_fase1 != start_epoch:
            epoch_consistente = min(start_epoch, epoch_csv_fase1)
            print(f"Aviso: inconsistencia entre CSV (epoca {epoch_csv_fase1}) e backup (epoca {start_epoch}).")
            print(f"Usando epoca {epoch_consistente} como ponto mais recente consistente.")
            # Remove linhas do historico posteriores ao backup para evitar duplicacao
            historico = [h for h in historico if h["epoch"] <= epoch_consistente]
            salvar_historico(csv_fase1, historico)
            # Reconstroi EarlyStopping a partir do CSV truncado
            if historico:
                df_csv = pd.DataFrame(historico)
                best_loss = float(df_csv["val_loss"].min())
                best_epoch_fase1 = int(df_csv.loc[df_csv["val_loss"].idxmin(), "epoch"])
                patience_count = int(((df_csv["epoch"] > best_epoch_fase1) & (df_csv["val_loss"] >= best_loss)).sum())
            else:
                best_loss = float("inf")
                best_epoch_fase1 = 0
                patience_count = 0
            start_epoch = epoch_consistente

        proxima_epoca = start_epoch + 1
        if patience_count >= PATIENCE:
            start_epoch = EPOCHS
            proxima_epoca = EPOCHS + 1
            print("Fase 1 ja finalizada por EarlyStopping")
        else:
            print(f"Resume fase 1: continuando da epoca {proxima_epoca}")
    elif os.path.exists(checkpoint_path):
        estado_ckpt = carregar_estado(checkpoint_path, model, optimizer, device=DEVICE, scheduler=scheduler)
        if estado_ckpt is not None:
            best_loss = float(estado_ckpt.get("best_loss", best_loss))
            best_epoch_fase1 = int(estado_ckpt.get("best_epoch", best_epoch_fase1))
            if historico:
                df_hist = pd.DataFrame(historico)
                mask = (df_hist["epoch"] > best_epoch_fase1) & (df_hist["val_loss"] >= best_loss)
                patience_count = int(mask.sum())
            else:
                patience_count = 0
            print("Checkpoint da fase 1 carregado (fallback). patience_count reconstruido do CSV.")
    else:
        print("Sem checkpoint anterior. Iniciando fase 1 do zero.", flush=True)

    inicio = time.time()

    # A variavel epoch no loop ja e o numero real da epoca (1, 2, 3, ...).
    for epoch in range(proxima_epoca, EPOCHS + 1):
        epoch_inicio = time.time()
        print(f"Iniciando epoch {epoch}/{EPOCHS}", flush=True)

        t_train_inicio = time.time()
        loss, acc, prec, rec, f1, auc = treinar_epoch(optimizer, scaler, use_amp)
        t_train = time.time() - t_train_inicio

        t_val_inicio = time.time()
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = avaliar_loader(val_loader)
        t_val = time.time() - t_val_inicio

        epoch_elapsed = time.time() - epoch_inicio
        remaining_epochs = EPOCHS - epoch
        eta_restante = epoch_elapsed * remaining_epochs

        print(
            f"Epoch {epoch} | "
            f"Loss {loss:.4f} | "
            f"Acc {acc:.4f} | "
            f"Prec {prec:.4f} | "
            f"Rec {rec:.4f} | "
            f"F1 {f1:.4f} | "
            f"AUC {auc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Prec {val_prec:.4f} | "
            f"Val Rec {val_rec:.4f} | "
            f"Val F1 {val_f1:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"tempo epoca {formatar_tempo(epoch_elapsed)} | "
            f"ETA restante {formatar_tempo(eta_restante)}"
        )

        lr_before_scheduler = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after_step = optimizer.param_groups[0]["lr"]
        scheduler_step = lr_after_step != lr_before_scheduler

        # Atualiza tracking de melhor modelo (operacao em memoria, segura)
        foi_melhor = val_loss < best_loss
        if foi_melhor:
            best_loss = val_loss
            best_epoch_fase1 = epoch
            patience_count = 0
        else:
            patience_count += 1

        if DEBUG_TIMING:
            t_ckpt_inicio = time.time()

        # Backup PRIMEIRO (antes do CSV) para garantir consistencia em caso de interrupcao
        salvar_estado(
            backup_fase1_path,
            model,
            optimizer,
            epoch,
            best_loss,
            patience_count,
            phase="fase1_backup",
            model_name=MODEL_NAME,
            best_epoch=best_epoch_fase1,
            scheduler_state_dict=scheduler.state_dict()
        )

        # Checkpoint apenas se foi melhora
        if foi_melhor:
            salvar_estado(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_loss,
                patience_count,
                phase="fase1_best",
                model_name=MODEL_NAME,
                best_epoch=best_epoch_fase1,
                scheduler_state_dict=scheduler.state_dict()
            )

        # CSV depois do backup
        if DEBUG_TIMING:
            t_csv_inicio = time.time()

        historico.append({
            "epoch": epoch,
            "accuracy": acc,
            "auc": auc,
            "f1_score": f1,
            "loss": loss,
            "lr": lr_before_scheduler,
            "scheduler_step": scheduler_step,
            "precision": prec,
            "recall": rec,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "val_f1_score": val_f1,
            "val_loss": val_loss,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "trainable_params": TRAINABLE_PARAMS,
        })
        salvar_historico(csv_fase1, historico)

        if DEBUG_TIMING:
            t_csv = time.time() - t_csv_inicio
            t_ckpt = time.time() - t_ckpt_inicio
            print(
                f"  TIMING: train={t_train:.1f}s | val={t_val:.1f}s | "
                f"csv={t_csv:.2f}s | ckpt={t_ckpt:.1f}s | "
                f"total={epoch_elapsed:.1f}s"
            )

        if patience_count >= PATIENCE:
            print("EarlyStopping")
            break

    # =====================================================
    # SEÇÃO 5: FINE-TUNING
    # =====================================================

    print("\nIniciando fine-tuning...\n")

    if os.path.exists(checkpoint_path):
        carregar_estado(checkpoint_path, model, device=DEVICE)
        print("Melhor modelo da fase 1 carregado para fine-tuning")

    num_ft_blocks = FINE_TUNE_BLOCKS.get(PREFIX, 2)
    liberar_fine_tuning(model, num_blocks=num_ft_blocks)
    TRAINABLE_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
    TRAINABLE_PARAMS_FT = TRAINABLE_PARAMS
    print(f"Parametros treinaveis fine-tuning: {TRAINABLE_PARAMS:,}")
    print(f"Blocos liberados no fine-tuning: {num_ft_blocks}")
    optimizer = criar_otimizador(model, lr=1e-5)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type) if use_amp else None
    scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_ft_loss = float("inf")
    best_epoch_ft = 0
    ft_patience_count = 0
    historico_ft = carregar_historico(csv_ft)
    start_epoch_ft = ultimo_epoch_csv(csv_ft)
    # proxima_epoca_ft: primeira epoca a treinar (1-indexed). Padrao para fluxos sem backup.
    proxima_epoca_ft = start_epoch_ft + 1

    estado_ft = carregar_estado(backup_ft_path, model, optimizer, device=DEVICE, scheduler=scheduler_ft)
    if estado_ft is not None:
        epoch_backup_ft = int(estado_ft.get("epoch", start_epoch_ft))
        start_epoch_ft = epoch_backup_ft
        best_ft_loss = float(estado_ft.get("best_loss", best_ft_loss))
        best_epoch_ft = int(estado_ft.get("best_epoch", best_epoch_ft))
        ft_patience_count = int(estado_ft.get("patience_count", ft_patience_count))

        epoch_csv_ft = ultimo_epoch_csv(csv_ft)
        if epoch_csv_ft > 0 and epoch_csv_ft != epoch_backup_ft:
            epoch_consistente = min(epoch_backup_ft, epoch_csv_ft)
            print(f"Aviso: inconsistência entre CSV (época {epoch_csv_ft}) e backup (época {epoch_backup_ft}).")
            print(f"Usando época {epoch_consistente} como ponto mais recente consistente.")
            # Remove linhas do historico posteriores ao backup para evitar duplicacao
            historico_ft = [h for h in historico_ft if h["epoch"] <= epoch_consistente]
            salvar_historico(csv_ft, historico_ft)
            # Reconstroi EarlyStopping a partir do CSV truncado
            if historico_ft:
                df_csv = pd.DataFrame(historico_ft)
                best_ft_loss = float(df_csv["val_loss"].min())
                best_epoch_ft = int(df_csv.loc[df_csv["val_loss"].idxmin(), "epoch"])
                ft_patience_count = int(((df_csv["epoch"] > best_epoch_ft) & (df_csv["val_loss"] >= best_ft_loss)).sum())
            else:
                best_ft_loss = float("inf")
                best_epoch_ft = 0
                ft_patience_count = 0
            start_epoch_ft = epoch_consistente

        proxima_epoca_ft = start_epoch_ft + 1
        if ft_patience_count >= FT_PATIENCE:
            start_epoch_ft = FINE_TUNE_EPOCHS
            proxima_epoca_ft = FINE_TUNE_EPOCHS + 1
            print("Fine-Tuning já finalizado por EarlyStopping")
        else:
            print(f"Resume fine-tuning: continuando da época {proxima_epoca_ft}")

    if estado_ft is None and os.path.exists(checkpoint_ft_path):
        estado_ckpt_ft = carregar_estado(checkpoint_ft_path, model, optimizer, device=DEVICE, scheduler=scheduler_ft)
        if estado_ckpt_ft is not None:
            best_ft_loss = float(estado_ckpt_ft.get("best_loss", best_ft_loss))
            best_epoch_ft = int(estado_ckpt_ft.get("best_epoch", best_epoch_ft))
            ft_patience_count = int(estado_ckpt_ft.get("patience_count", ft_patience_count))
            if historico_ft:
                df_hist_ft = pd.DataFrame(historico_ft)
                mask = (df_hist_ft["epoch"] > best_epoch_ft) & (df_hist_ft["val_loss"] >= best_ft_loss)
                ft_patience_count = int(mask.sum())
            print("Backup do Fine-Tuning nao encontrado. Checkpoint carregado com optimizer/scheduler.")

    # A variavel epoch no loop ja e o numero real da epoca (1, 2, 3, ...).
    for epoch in range(proxima_epoca_ft, FINE_TUNE_EPOCHS + 1):
        ft_inicio = time.time()
        print(f"Iniciando fine-tuning {epoch}/{FINE_TUNE_EPOCHS}", flush=True)

        loss, acc, prec, rec, f1, auc = treinar_epoch(optimizer, scaler, use_amp)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = avaliar_loader(val_loader)
        ft_elapsed = time.time() - ft_inicio
        ft_remaining = (FINE_TUNE_EPOCHS - epoch) * ft_elapsed

        print(
            f"Fine-tuning {epoch} | "
            f"Loss {loss:.4f} | "
            f"Acc {acc:.4f} | "
            f"Prec {prec:.4f} | "
            f"Rec {rec:.4f} | "
            f"F1 {f1:.4f} | "
            f"AUC {auc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Prec {val_prec:.4f} | "
            f"Val Rec {val_rec:.4f} | "
            f"Val F1 {val_f1:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"tempo epoca {formatar_tempo(ft_elapsed)} | "
            f"ETA restante {formatar_tempo(ft_remaining)}"
        )

        lr_before_scheduler = optimizer.param_groups[0]["lr"]
        scheduler_ft.step(val_loss)
        scheduler_step_ft = optimizer.param_groups[0]["lr"] != lr_before_scheduler

        # Atualiza tracking de melhor modelo
        foi_melhor_ft = val_loss < best_ft_loss
        if foi_melhor_ft:
            best_ft_loss = val_loss
            best_epoch_ft = epoch
            ft_patience_count = 0
        else:
            ft_patience_count += 1

        # Backup PRIMEIRO (antes do CSV)
        salvar_estado(
            backup_ft_path,
            model,
            optimizer,
            epoch,
            best_ft_loss,
            ft_patience_count,
            phase="finetuning_backup",
            model_name=MODEL_NAME,
            best_epoch=best_epoch_ft,
            scheduler_state_dict=scheduler_ft.state_dict()
        )

        # Checkpoint apenas se foi melhora
        if foi_melhor_ft:
            salvar_estado(
                checkpoint_ft_path,
                model,
                optimizer,
                epoch,
                best_ft_loss,
                ft_patience_count,
                phase="finetuning_best",
                model_name=MODEL_NAME,
                best_epoch=best_epoch_ft,
                scheduler_state_dict=scheduler_ft.state_dict()
            )

        # CSV depois do backup
        historico_ft.append({
            "epoch": epoch,
            "accuracy": acc,
            "auc": auc,
            "f1_score": f1,
            "loss": loss,
            "lr": lr_before_scheduler,
            "scheduler_step": scheduler_step_ft,
            "precision": prec,
            "recall": rec,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "val_f1_score": val_f1,
            "val_loss": val_loss,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "trainable_params": TRAINABLE_PARAMS,
        })
        salvar_historico(csv_ft, historico_ft)

        if ft_patience_count >= FT_PATIENCE:
            print("Fine-Tuning EarlyStopping")
            break

    # =====================================================
    # SEÇÃO 6: SALVAMENTO
    # =====================================================

    # =====================================================
    # CARREGAR MELHOR CHECKPOINT DO FINE-TUNING
    # =====================================================

    if os.path.exists(checkpoint_ft_path):
        print(f"Carregando melhores pesos do fine-tuning: {checkpoint_ft_path}")
        carregar_estado(checkpoint_ft_path, model, device=DEVICE)
    elif os.path.exists(checkpoint_path):
        carregar_estado(checkpoint_path, model, device=DEVICE)

    # =====================================================
    # SALVAR MODELO FINAL
    # =====================================================
    torch.save({
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
        "positive_class_name": POSITIVE_CLASS_NAME,
        "positive_idx": positive_idx,
        "num_classes": NUM_CLASSES,
        "input_size": IMG_SIZE,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }, final_path)

    # =====================================================
    # LIMPEZA DE ARQUIVOS TEMPORÁRIOS DE BACKUP
    # =====================================================
    for temp_file in [backup_fase1_path, backup_ft_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"  Removido: {temp_file}")

    fim = time.time()

    # =====================================================
    # SEÇÃO 7: FINALIZAÇÃO
    # =====================================================

    tempo_total_min = (fim - inicio) / 60
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {MODEL_NAME}\n")
        f.write(f"Parametros totais: {PARAMS:,}\n")
        f.write(f"Parametros treinaveis (Fase 1): {TRAINABLE_PARAMS_FASE1:,}\n")
        f.write(f"Parametros treinaveis (Fine-Tuning): {TRAINABLE_PARAMS_FT:,}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Workers: {num_workers}\n")
        f.write(f"Learning Rate (Fase 1): 1e-4\n")
        f.write(f"Learning Rate (Fine-Tuning): 1e-5\n")
        f.write(f"Scheduler: ReduceLROnPlateau (patience=5 fase1, patience=3 ft)\n")
        f.write(f"FINE_TUNE_BLOCKS: {num_ft_blocks}\n")
        f.write(f"Classes: {train_ds.class_to_idx}\n")
        f.write(f"Classe positiva: {POSITIVE_CLASS_NAME} ({positive_idx})\n")
        f.write(f"\n")
        f.write(f"--- Fase 1 ---\n")
        f.write(f"Melhor Validation Loss: {best_loss:.6f}\n")
        f.write(f"Epoca do melhor: {best_epoch_fase1}\n")
        f.write(f"\n")
        f.write(f"--- Fine-Tuning ---\n")
        f.write(f"Melhor Validation Loss: {best_ft_loss:.6f}\n")
        f.write(f"Epoca do melhor: {best_epoch_ft}\n")
        f.write(f"\n")
        f.write(f"Tempo total de treinamento (min): {tempo_total_min:.2f}\n")

    # =====================================================
    # CONFIG.JSON
    # =====================================================
    config_path = os.path.join(modelos_dir, f"{PREFIX}_config.json")
    config = {
        "arquitetura": MODEL_NAME,
        "epochs": EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "patience": PATIENCE,
        "fine_tuning_patience": FT_PATIENCE,
        "fine_tune_blocks": num_ft_blocks,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "learning_rates": {"fase1": 1e-4, "finetuning": 1e-5},
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": {"fase1": 5, "finetuning": 3},
        "scheduler_factor": 0.5,
        "optimizer": "Adam",
        "class_weight": usar_class_weight,
        "seed": SEED,
        "device": str(DEVICE),
        "num_workers": num_workers,
        "data_augmentation": [
            "RandomRotation(10)",
            "ColorJitter(brightness, contrast, saturation)",
            "RandomResizedCrop"
        ],
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "numpy_version": np.__version__
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(converter_para_json(config), f, indent=2)
    print(f"  Config salvo: {config_path}")

    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main()
