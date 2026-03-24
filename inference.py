import argparse, torch, numpy as np, cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from model import StudentModel

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except Exception:
    HAS_CRF = False


def preprocess(img_path, img_size):
    im = Image.open(img_path).convert("RGB")
    im = TF.resize(im, [img_size, img_size], antialias=True)
    return TF.to_tensor(im).unsqueeze(0), np.array(im)


def apply_crf(image, prob):
    if not HAS_CRF:
        return prob

    h, w = image.shape[:2]
    unary = unary_from_softmax(np.stack([1 - prob, prob], axis=0))

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(3, 3)
    d.addPairwiseBilateral(60, 13, image, 10)

    Q = np.array(d.inference(5))     # <-- FIXED
    return Q[1].reshape(h, w)


def postprocess(prob, thr=0.5):
    m = (prob >= thr).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (labels == largest).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return m * 255

def overlay(original, mask):
    # Ensure original is 3-channel
    if len(original.shape) == 2:  # grayscale → convert
        orig_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:  # already RGB
        orig_color = original

    color_mask = np.stack([mask*0, mask, mask*0], axis=-1).astype(np.uint8)
    return cv2.addWeighted(orig_color, 0.6, color_mask, 0.4, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--in_dir", default=None)
    ap.add_argument("--out_dir", default="./preds")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location=device)

    # Your StudentModel likely expects no constructor args
    model = StudentModel().to(device)

    # When saving, you saved ONLY state_dict — so handle both formats
    if "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
    else:
        model.load_state_dict(ck)

    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.in_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths += list(Path(args.in_dir).glob(ext))

    assert paths, "❗ Provide --image or --in_dir"

    for p in paths:
        x, orig = preprocess(p, args.img_size)
        x = x.to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        if HAS_CRF:
            prob = apply_crf(orig, prob)

        mask = postprocess(prob, args.thr)

        Image.fromarray(mask).save(out / f"{p.stem}_mask.png")
        cv2.imwrite(str(out / f"{p.stem}_overlay.png"), overlay(orig, mask))

    print(f"Results saved to {out} ({'with CRF' if HAS_CRF else 'no CRF'})")


if __name__ == "__main__":
    main()
