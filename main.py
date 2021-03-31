import sys
from pathlib import Path
import argparse

import torch
import torchvision
from PIL import Image

from vrb_model import vrb_full_model
from scripts import load_model, process_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path")
    parser.add_argument("--save-path", default=Path(r"detections"),
                        help="The folder to save the visual relationship detections.")
    parser.add_argument("--model-path", default=Path(r"pretrained_models/optimal_model/optimal_model.tar"))

    args = parser.parse_args()

    model_vrb, device = load_model(args.model_path)
    process_image(args.img_path, model_vrb, device, args.save_path)
