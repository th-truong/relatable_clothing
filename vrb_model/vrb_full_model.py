from vrb_model import vrb_training_model, rcnn

from collections import OrderedDict
import warnings
import itertools

from torch.jit.annotations import Tuple, List, Dict, Optional
import torch
from torch import nn

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def create_full_vrb_model(num_classes, model_mrcnn, model_path=None,
                     pool_sizes={'0': (50, 50),
                                 '1': (25, 25),
                                 '2': (12, 12),
                                 '3': (12, 12),
                                 'pool': (6, 6)},
                     mask_mold_filters=8,
                     soft_attention_filters=256,
                     bottleneck_filters=16,
                     hidden_layer_sizes=[128, 64, 128],
                     mask_mold_ablation=False
                     ):

    mask_transform = vrb_training_model.MaskTransformVRB(800, 1333)

    FPN_filters = 256 # this is the default number of filters on the FPN output for each level

    mask_mold = vrb_training_model.MoldMaskInputsVRB(mask_mold_filters=mask_mold_filters,
                                                     mask_mold_ablation=mask_mold_ablation)

    soft_attention = vrb_training_model.SoftAttentionMechanismVRB(FPN_filters=FPN_filters, mask_mold_filters=mask_mold_filters,
                                               soft_attention_filters=soft_attention_filters,
                                               bottleneck_filters=bottleneck_filters,
                                               keys=list(pool_sizes.keys()),
                                               mask_mold_ablation=mask_mold_ablation)

    classifier_head = vrb_training_model.MLPClassifierHead(pool_sizes, num_classes=num_classes,
                                        hidden_layer_sizes=hidden_layer_sizes, bottleneck_filters=bottleneck_filters,
                                        keys=list(pool_sizes.keys()))

    model = VRBFullModel(model_mrcnn, mask_transform, mask_mold, soft_attention, classifier_head)

    return model


def create_mrcnn_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    model_mrcnn = MaskRCNNVRB(backbone, num_classes)

    # get number of input features for the classifier
    in_features = model_mrcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model_mrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model_mrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model_mrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
    return model_mrcnn


class VRBFullModel(nn.Module):

    def __init__(self, model_mrcnn, mask_transform, mask_mold, soft_attention, classifier_head,
                 obj_mask_threshold=0.5, person_mask_threshold=0.8, score_threshold=0.25,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(VRBFullModel, self).__init__()
        self.model_mrcnn = model_mrcnn
        self.mask_transform = mask_transform
        self.mask_mold = mask_mold
        self.soft_attention = soft_attention
        self.classifier_head = classifier_head

        self.obj_mask_threshold = obj_mask_threshold
        self.person_mask_threshold = person_mask_threshold
        self.score_threshold = score_threshold
        self.device = device
        self.device_float_one = torch.Tensor([1.0]).to(self.device)
        self.device_float_zero = torch.Tensor([0.0]).to(self.device)

        self.catID_to_label = catID_to_label = {1: 'backpack',
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

    def forward(self, img, targets=None):
        losses, detections, features = self.model_mrcnn(img)

        detections = detections[0]

        person_mask_indices = [i for i in range(len(detections['scores']))
                               if (detections['labels'][i] == 4
                               or detections['labels'][i] == 8)
                               and (detections['scores'][i] >= self.score_threshold)]

        obj_mask_indices = [i for i in range(len(detections['scores']))
                            if not (detections['labels'][i] == 4
                            or detections['labels'][i] == 8)
                            and (detections['scores'][i] >= self.score_threshold)]

        person_obj_pairs = [pair for pair in itertools.product(person_mask_indices,
                                                               obj_mask_indices)]

        predictions = []

        for i, pair in enumerate(person_obj_pairs):
            person = detections['masks'][pair[0]]
            person = torch.where(person >= self.person_mask_threshold,
                                 self.device_float_one, self.device_float_zero)
            person = [person]
            obj = detections['masks'][pair[1]]
            obj = torch.where(obj >= self.obj_mask_threshold,
                              self.device_float_one, self.device_float_zero)
            obj = [obj]
            person_image_list, obj_image_list, targets = self.mask_transform(person, obj, targets)

            soft_mask_input, targets = self.mask_mold(person_image_list, obj_image_list, targets)

            attention_features, targets = self.soft_attention(features, soft_mask_input, targets)

            prediction, targets = self.classifier_head(attention_features, targets)

            predictions.append(prediction)

        return detections, predictions, person_obj_pairs, targets


class MaskRCNNVRB(MaskRCNN):
    def __init__(self, backbone, num_classes=None):
        # modifies the MaskRCNN forward function to also return the backbone features
        super(MaskRCNNVRB, self).__init__(backbone, num_classes)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

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
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return (losses, detections, features)
