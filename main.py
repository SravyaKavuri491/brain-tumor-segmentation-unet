# ============================================================
# main.py — Balanced U-Net Trainer (Hybrid Loss, scSE Attention)
# with Early Stopping + LR Scheduler
# ============================================================
import os, random, numpy as np, torch, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from dataset import SegDataset
from model import StudentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./runs", exist_ok=True)

# ============================================================
# 🔹 Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return

        if self.mode == 'max':
            improved = metric > (self.best + self.min_delta)
        else:
            improved = metric < (self.best - self.min_delta)

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# 🔹 Loss Functions
# ============================================================
def weighted_bce_dice(pred, target, pos_weight=3.0):
    bce = F.binary_cross_entropy_with_logits(
        pred, target,
        pos_weight=torch.tensor(pos_weight, device=pred.device)
    )
    p = torch.sigmoid(pred)
    smooth = 1.
    inter = (p * target).sum()
    dice = 1 - ((2 * inter + smooth) / (p.sum() + target.sum() + smooth))
    return 0.6 * bce + 0.4 * dice


def focal_bce(pred, target, gamma=2.0):
    p = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.where(target == 1, p, 1 - p)
    return (((1 - pt) ** gamma) * bce).mean()


def hybrid_loss(pred, target):
    return 0.4 * focal_bce(pred, target) + 0.6 * weighted_bce_dice(pred, target)


def dice_coef(pred, target):
    p = (torch.sigmoid(pred) > 0.5).float()
    smooth = 1.
    return (2 * (p * target).sum() + smooth) / (p.sum() + target.sum() + smooth)


# ============================================================
# 🔹 Training Function
# ============================================================
def train(train_csv, val_csv, root="./ADDA", epochs=50, batch_size=8, lr=1e-3):

    # Dataset
    tr_ds = SegDataset(train_csv, root, augment=True)
    va_ds = SegDataset(val_csv, root, augment=False)

    # ------------------------------------------------------------
    # Balanced Sampling
    # ------------------------------------------------------------
    df = pd.read_csv(train_csv)

    tumor_idx = []
    for i, r in df.iterrows():
        mask_path = os.path.join(root, str(r["label_path"]))
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.sum() > 0:
                tumor_idx.append(i)
        except:
            pass

    non_tumor_idx = [i for i in range(len(df)) if i not in tumor_idx]

    balanced_idx = tumor_idx * 2 + random.sample(
        non_tumor_idx,
        min(len(non_tumor_idx), len(tumor_idx) * 2)
    )

    sampler = SubsetRandomSampler(balanced_idx)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ------------------------------------------------------------
    # Model + Optimizer
    # ------------------------------------------------------------
    model = StudentModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=5, factor=0.5
    )

    early_stop = EarlyStopping(patience=10, mode='max')

    best, trL, vaD = 0., [], []

    print(f"\n🚀 Starting training for {epochs} epochs\n")

    # ============================================================
    # Epoch Loop
    # ============================================================
    for e in range(epochs):
        model.train()
        runL = 0.

        loop = tqdm(tr_ld, desc=f"Epoch {e+1}/{epochs} [Train]", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            p = model(x)
            L = hybrid_loss(p, y)
            L.backward()
            opt.step()

            runL += L.item() * x.size(0)
            loop.set_postfix(loss=L.item())

        tr_loss = runL / len(tr_ld.dataset)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_d = 0.

        vloop = tqdm(va_ld, desc=f"Epoch {e+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for x, y in vloop:
                x, y = x.to(device), y.to(device)
                p = model(x)
                d = dice_coef(p, y).item()
                val_d += d * x.size(0)
                vloop.set_postfix(dice=d)

        val_d /= len(va_ld.dataset)
        sched.step(val_d)

        print(f"Epoch {e+1:03d}/{epochs} | TrainLoss={tr_loss:.4f} | ValDice={val_d:.4f}")

        # Save best model
        if val_d > best:
            best = val_d
            torch.save(model.state_dict(), "./runs/best_unet_balanced.pth")
            print(f"✅ Saved best model (Dice={best:.4f})")

        trL.append(tr_loss)
        vaD.append(val_d)

        # ----------------------------
        # Early Stopping Check
        # ----------------------------
        early_stop(val_d)
        if early_stop.stop:
            print("⛔ Early stopping triggered!")
            break

    # ------------------------------------------------------------
    # Plot Curves
    # ------------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(trL, label="Train Loss")
    plt.plot(vaD, label="Val Dice")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Training Progress")
    plt.savefig("./runs/training_curve.png")
    plt.show()

    print(f"\n🎯 Final Best Dice={best:.4f}")
    print("📌 Model saved at: ./runs/best_unet_balanced.pth")
    print("📈 Curve saved at: ./runs/training_curve.png")


# ============================================================
# 🔹 CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./ADDA")
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--val_csv", type=str, default="val.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train(
        os.path.join(args.data_dir, args.train_csv),
        os.path.join(args.data_dir, args.val_csv),
        root=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
