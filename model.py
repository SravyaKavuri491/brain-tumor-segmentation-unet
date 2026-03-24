# ============================================================
# model.py — U-Net (ResNet34 + scSE Attention)
# ============================================================
import torch
import segmentation_models_pytorch as smp

class StudentModel(torch.nn.Module):
    def __init__(self, encoder_name="resnet34", pretrained=True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse"  # CBAM-like channel+spatial squeeze excitation
        )

    def forward(self, x):
        return self.net(x)
