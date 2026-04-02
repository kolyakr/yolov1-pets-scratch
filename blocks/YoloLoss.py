from utils import intersection_over_union
import torch
from torch import nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=37):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, preds, targets):
        exists_box = targets[..., 4:5]
        no_box_exists = 1 - exists_box

        preds_box1 = preds[..., 0:4]
        preds_box2 = preds[..., 5:9]
        targets_box = targets[..., 0:4]

        preds_box1_iou = intersection_over_union(preds_box1, targets_box)
        preds_box2_iou = intersection_over_union(preds_box2, targets_box)

        box1_winner = (preds_box1_iou >= preds_box2_iou).to(int)
        box2_winner = (preds_box1_iou < preds_box2_iou).to(int)

        preds_xy_1 = preds[..., 0:2]
        preds_xy_2 = preds[..., 5:7]
        targets_xy = targets[..., 0:2]

        loss_xy_1 = torch.pow(targets_xy - preds_xy_1, 2)
        loss_xy_2 = torch.pow(targets_xy - preds_xy_2, 2)

        final_xy_loss = exists_box * ((loss_xy_1 * box1_winner) + (loss_xy_2 * box2_winner))
        total_xy_loss = self.lambda_coord * torch.sum(final_xy_loss)

        preds_wh_1 = torch.sign(preds[..., 2:4]) * torch.sqrt(torch.abs(preds[..., 2:4]) + 1e-6)
        preds_wh_2 = torch.sign(preds[..., 7:9]) * torch.sqrt(torch.abs(preds[..., 7:9]) + 1e-6)
        targets_wh = torch.sqrt(targets[..., 2:4])

        loss_wh_1 = torch.pow(targets_wh - preds_wh_1, 2)
        loss_wh_2 = torch.pow(targets_wh - preds_wh_2, 2)

        final_wh_loss = exists_box * ((loss_wh_1 * box1_winner) + (loss_wh_2 * box2_winner))
        total_wh_loss = self.lambda_coord * torch.sum(final_wh_loss)

        preds_conf_1 = torch.sigmoid(preds[..., 4:5])
        preds_conf_2 = torch.sigmoid(preds[..., 9:10])
        
        targets_conf_1 = preds_box1_iou.detach()
        targets_conf_2 = preds_box2_iou.detach()

        loss_obj_1 = torch.pow(targets_conf_1 - preds_conf_1, 2)
        loss_obj_2 = torch.pow(targets_conf_2 - preds_conf_2, 2)

        final_obj_loss = exists_box * ((loss_obj_1 * box1_winner) + (loss_obj_2 * box2_winner))
        total_obj_loss = torch.sum(final_obj_loss)


        loss_noobj_1 = torch.pow(0.0 - preds_conf_1, 2)
        loss_noobj_2 = torch.pow(0.0 - preds_conf_2, 2)

        final_noobj_loss = (
            (no_box_exists * (loss_noobj_1 + loss_noobj_2)) + 
            (exists_box * ((loss_noobj_1 * box2_winner) + (loss_noobj_2 * box1_winner)))
        )
        total_noobj_loss = self.lambda_noobj * torch.sum(final_noobj_loss)

        preds_classes = torch.softmax(preds[..., 10:], dim=-1)
        targets_classes = targets[..., 10:]

        loss_classes = torch.pow(targets_classes - preds_classes, 2)

        final_class_loss = exists_box * loss_classes
        total_class_loss = torch.sum(final_class_loss)


        total_loss = (
            total_xy_loss + 
            total_wh_loss + 
            total_obj_loss + 
            total_noobj_loss + 
            total_class_loss
        )

        return total_loss, total_xy_loss, total_wh_loss, \
               total_obj_loss, total_noobj_loss, total_class_loss