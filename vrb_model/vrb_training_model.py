"""
"""
from torch.jit.annotations import Tuple, List, Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from collections import OrderedDict

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision

import math
import numpy as np

from vrb_model import rcnn


def trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def create_vrb_model(num_classes, model_path=None, backbone_arch='resnet50',
                     pool_sizes={'0': (20, 20),
                                 '1': (20, 20),
                                 '2': (20, 20),
                                 '3': (20, 20),
                                 'pool': (20, 20)},
                     mask_mold_filters=64,
                     soft_attention_filters=256,
                     bottleneck_filters=32,
                     hidden_layer_sizes=[256, 128, 256],
                     mask_mold_ablation=False
                     ):
    mask_transform = MaskTransformVRB(800, 1333)

    img_transform_backbone = load_VRB_backbonefpn_weights(model_path, backbone_arch=backbone_arch)
    # freeze all backbone layers
    for param in img_transform_backbone.parameters():
        param.requires_grad = False

    FPN_filters = 256 # this is the default number of filters on the FPN output for each level

    mask_mold = MoldMaskInputsVRB(mask_mold_filters=mask_mold_filters,
                                  mask_mold_ablation=mask_mold_ablation)

    soft_attention = SoftAttentionMechanismVRB(FPN_filters=FPN_filters, mask_mold_filters=mask_mold_filters,
                                               soft_attention_filters=soft_attention_filters,
                                               bottleneck_filters=bottleneck_filters,
                                               keys=list(pool_sizes.keys()),
                                               mask_mold_ablation=mask_mold_ablation)

    classifier_head = MLPClassifierHead(pool_sizes, num_classes=num_classes,
                                        hidden_layer_sizes=hidden_layer_sizes, bottleneck_filters=bottleneck_filters,
                                        keys=list(pool_sizes.keys()))

    model = VRBTrainingModel(img_transform_backbone, mask_transform, mask_mold, soft_attention, classifier_head)

    print(f"There are {str(trainable_params(model))} trainable parameters.")

    return model


class VRBTrainingModel(nn.Module):

    def __init__(self, img_transform_backbone, mask_transform, mask_mold, soft_attention, classifier_head):
        super(VRBTrainingModel, self).__init__()
        self.backbone = img_transform_backbone
        self.mask_transform = mask_transform
        self.mask_mold = mask_mold
        self.soft_attention = soft_attention
        self.classifier_head = classifier_head

    def forward(self, img, person, obj, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        features = self.backbone(img, targets)
        person_image_list, obj_image_list, targets = self.mask_transform(person, obj, targets)

        soft_mask_input, targets = self.mask_mold(person_image_list, obj_image_list, targets)

        attention_features, targets = self.soft_attention(features, soft_mask_input, targets)

        predictions, targets = self.classifier_head(attention_features, targets)

        return predictions, targets


def load_VRB_backbonefpn_weights(model_path, min_size=800, max_size=1333,
                                 image_mean=None, image_std=None, backbone_arch="resnet50"):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    transform = rcnn.GeneralizedRCNNTransformVRB(min_size, max_size, image_mean, image_std)

    backbone = resnet_fpn_backbone(backbone_arch, False, trainable_layers=5)

    model = TransformBackboneRPNVRB(backbone, transform)

    if model_path is not None:
        checkpoint = torch.load(model_path)
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)

    return model


class MLPClassifierHead(nn.Module):

    def __init__(self, pool_sizes, num_classes, hidden_layer_sizes=[256, 128, 256],
                 bottleneck_filters=32, keys=['0', '1', '2', '3', 'pool']):
        super(MLPClassifierHead, self).__init__()

        self.keys = keys

        num_features = 0
        for pool_size in pool_sizes.values():
            area = pool_size[0] * pool_size[1]
            num_features += area * bottleneck_filters

        self.num_features = num_features

        self.avg_pool = nn.ModuleDict({key: nn.AdaptiveAvgPool2d(pool_sizes[key]) for key in self.keys})

        self.fc0 = nn.Linear(self.num_features, hidden_layer_sizes[0])
        self.fc0dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc1dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])

        self.fc_classifier = nn.Linear(hidden_layer_sizes[2], num_classes)

    def forward(self, attention_features, targets):
        pool_outs = [self.avg_pool[key](attention_features[key])
                     for key in self.keys]

        pool_outs = [pool_out.flatten(start_dim=1) for pool_out in pool_outs]

        concat = torch.cat(tuple(pool_outs), 1)

        fc0_out = F.relu(self.fc0(concat))
        fc0drop_out = self.fc0dropout(fc0_out)

        fc1_out = F.relu(self.fc1(fc0drop_out))
        fc1drop_out = self.fc1dropout(fc1_out)

        fc2_out = F.relu(self.fc2(fc1drop_out))

        x = torch.sigmoid(self.fc_classifier(fc2_out))

        return x, targets


class SoftAttentionMechanismVRB(nn.Module):
    def __init__(self, FPN_filters=256, mask_mold_filters=64, soft_attention_filters=256, bottleneck_filters=32,
                 keys=['0', '1', '2', '3', 'pool'], mask_mold_ablation=False):
        # keys must be a subset (or the entire set) of the keys in features from the FPN
        super(SoftAttentionMechanismVRB, self).__init__()
        self.mask_mold_ablation = mask_mold_ablation
        self.FPN_filters = FPN_filters
        if not self.mask_mold_ablation:
            self.mask_mold_filters = mask_mold_filters
        else:
            self.mask_mold_filters = 3
        self.soft_attention_filters = soft_attention_filters
        self.bottleneck_filters = bottleneck_filters
        self.keys = keys

        self.conv_layers = nn.ModuleDict({key: nn.Conv2d(self.mask_mold_filters, self.soft_attention_filters, kernel_size=3, padding=1) for key in keys})
        self.relu = nn.ModuleDict({key: nn.ReLU(inplace=True) for key in keys})

        self.bottleneck_conv = nn.ModuleDict({key: nn.Conv2d(self.FPN_filters + self.soft_attention_filters,
                                                             self.bottleneck_filters, kernel_size=1, bias=False) for key in keys})
        self.bottleneck_relu = nn.ModuleDict({key: nn.ReLU(inplace=True) for key in keys})

    def forward(self, features, soft_mask_input, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        feature_shapes = {key: features[key].shape[-2:] for key in self.keys}
        # for resnet50fpn feature_shapes contains (batch, filters, height, width) for corresponding feature key
        # the default for resnet50fpn is filters=256 and 5 keys in the features.

        masks_reshaped = {key: self._resize_soft_mask(soft_mask_input, feature_shapes[key])
                          for key in self.keys}

        masks_reshaped_filtered = {key: self.conv_layers[key](masks_reshaped[key])
                                    for key in self.keys}

        relu_out = {key: self.relu[key](masks_reshaped_filtered[key])
                    for key in self.keys}

        attention_features = {key: torch.cat((relu_out[key], features[key]), 1)
                              for key in self.keys}

        attention_features_bottleneck = {key: self.bottleneck_conv[key](attention_features[key])
                                         for key in self.keys}
        attention_features_bottleneck_relu = {key: self.bottleneck_relu[key](attention_features_bottleneck[key])
                                              for key in self.keys}

        return attention_features_bottleneck_relu, targets


    def _resize_soft_mask(self, mask, shapes):
        reshaped_mask = F.interpolate(mask.float(), size=(shapes[0], shapes[1]))

        return reshaped_mask


class MoldMaskInputsVRB(nn.Module):

    def __init__(self, mask_mold_filters=64, mask_mold_ablation=False):
        super(MoldMaskInputsVRB, self).__init__()
        # TODO: probably significantly reduce the number of inplanes here to reduce params significantly
        self.mask_mold_ablation = mask_mold_ablation
        self.mask_mold_filters = mask_mold_filters
        self.conv1 = nn.Conv2d(3, self.mask_mold_filters, kernel_size=7, stride=2, padding=3,
                               bias=False)

        if not self.mask_mold_ablation:
            self.bn1 = nn.BatchNorm2d(self.mask_mold_filters)  # probably remove this batch norm
        else:
            self.bn1 = nn.BatchNorm2d(3)  # probably remove this batch norm
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)




    def forward(self, person_image_list, obj_image_list, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        combined_mask = torch.add(person_image_list.tensors, obj_image_list.tensors)

        soft_mask = torch.cat((person_image_list.tensors, obj_image_list.tensors, combined_mask), 1)
        if not self.mask_mold_ablation:
            x = self.conv1(soft_mask.float())
        else:
            x = soft_mask.float()
        x = self.bn1(x)
        x = self.relu(x)
        soft_mask_input = self.maxpool(x)

        return soft_mask_input, targets


class TransformBackboneRPNVRB(nn.Module):  # taken from torchvision/models/detection/generalized_rcnn.py

    def __init__(self, backbone, transform):
        super(TransformBackboneRPNVRB, self).__init__()
        self.transform = transform
        self.backbone = backbone

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): ***

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenrate box
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invaid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        return features


class MaskTransformVRB(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNNVRB
    model.

    The transformations it perform are:
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size):
        super(MaskTransformVRB, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def forward(self,
                person_inputs,       # type: List[Tensor]
                obj_inputs,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        person_image_list = self.process_inputs(person_inputs)
        obj_image_list = self.process_inputs(obj_inputs)
        return person_image_list, obj_image_list, targets

    def process_inputs(self, images):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]

            image = self.resize(image)
            images[i] = image

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return(image_list)

    def resize(self, image):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        h, w = image.shape[-2:]
        size = float(self.min_size[-1])

        if torchvision._is_tracing():
            image = _resize_image_and_masks_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image_and_masks(image, size, float(self.max_size))

        return image

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs


@torch.jit.unused
def _resize_image_and_masks_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = F.interpolate(image[:, None].float(), scale_factor=scale_factor)[:, 0].byte()

    return image


def _resize_image_and_masks(image, self_min_size, self_max_size):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    image = F.interpolate(image[:, None].float(), scale_factor=scale_factor)[:, 0].byte()

    return image
