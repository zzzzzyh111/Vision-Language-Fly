import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from torchvision import transforms

from typing import List

from vlfly import IMAGE_ASPECT_RATIO, VLFly


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> VLFly:
    if config["model_type"] != "vlfly":
        raise ValueError(f"Invalid model type: {config['model_type']}")

    model = VLFly(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
        except AttributeError:
            state_dict = loaded_model.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return PILImage.fromarray(img)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False
) -> torch.Tensor:
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_imgs.append(torch.unsqueeze(transf_img, 0))
    return torch.cat(transf_imgs, dim=1)


def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
