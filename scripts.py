from pathlib import Path
from PIL import Image

import torch
import numpy as np
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import matplotlib

from vrb_model import vrb_full_model

def load_model(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    chkpt_full = torch.load(Path("pretrained_models/optimal_model/optimal_model.tar"))

    model_mrcnn = vrb_full_model.create_mrcnn_model(num_classes=22)
    model_vrb = vrb_full_model.create_full_vrb_model(num_classes=1, model_mrcnn=model_mrcnn)

    model_vrb.load_state_dict(chkpt_full['model'])
    model_vrb.eval()
    model_vrb.to(device)

    return model_vrb, device


def process_image(img_path, model_vrb, device, save_path):
    if not isinstance(img_path, Path):
        img_path = Path(img_path)

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = F.to_tensor(img)
    img = img.to(device)

    catID_to_label = {1: 'backpack',
                      2: 'belt',
                      3: 'dress',
                      4: 'female',
                      5: 'glove',
                      6: 'hat',
                      7: 'jeans',
                      8: 'male',
                      9: 'outerwear',
                      10: 'scarf',
                      11: 'shirt',
                      12: 'shoe',
                      13: 'shorts',
                      14: 'skirt',
                      15: 'sock',
                      16: 'suit',
                      17: 'swim_cap',
                      18: 'swim_wear',
                      19: 'tanktop',
                      20: 'tie',
                      21: 'trousers'}


    matplotlib.use('qt5agg')

    model_vrb.eval()
    model_vrb.to(device)
    with torch.no_grad():
        out = model_vrb([img])
    labels = out[0]['labels']
    masks = out[0]['masks']

    for i, pair in enumerate(out[2]):
        person_id = labels[pair[0]].cpu().detach().numpy()
        obj_id = labels[pair[1]].cpu().detach().numpy()
        person = catID_to_label[int(person_id)]
        obj = catID_to_label[int(obj_id)]
        vrb_score = out[1][i].cpu().detach().numpy()

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(masks[pair[0]].cpu().detach().numpy().squeeze())
        axes[1].imshow(masks[pair[1]].cpu().detach().numpy().squeeze())
        axes[0].set_title(f"{person}")

        if vrb_score > 0.5:
            obj_title = f"wearing {obj}"
        else:
            obj_title = f"not_wearing_{obj}"

        axes[1].set_title(obj_title)
        file_name = f"{person}_{obj_title}_{i}.png"
        plt.savefig(save_path/file_name, dpi=100, bbox_inches='tight', pad_inches=0)
