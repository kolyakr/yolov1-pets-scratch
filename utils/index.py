import os
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch
from blocks import DetectionHead, CatDogDataset
from torch import nn
import json

def parse_all_xmls(xml_folder_path):
    all_image_data = []
    
    for filename in os.listdir(xml_folder_path):
        if not filename.endswith(".xml"):
            continue 
            
        xml_path = os.path.join(xml_folder_path, filename)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        img_width = float(size.find("width").text)
        img_height = float(size.find("height").text)
    
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
        image_data = {
            "image_filename": filename.replace(".xml", ".jpg"), 
            "width": img_width,
            "height": img_height,
            "boxes": boxes,
            "labels": labels
        }
        
        all_image_data.append(image_data)
        
    return all_image_data

def get_detection_model(num_classes, freeze_backbone=True):
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False

    resnet = nn.Sequential(*list(resnet.children())[:-2])

    CatDogNet = nn.Sequential(
        *list(resnet.children()),
        DetectionHead(
            in_channels=512, 
            shrink_channels=128, 
            expand_channels=1024,
            S=7,
            B=2,
            num_classes=num_classes
        )
    )

    return CatDogNet

def get_data_loaders(batch_size=64, img_width=448, img_height=448, S=7, B=2):
    with open("../data/dataset_splits.json", "r") as f:
        dataset_splits = json.load(f)

    train_dataset = CatDogDataset(dataset_splits["train"], "../data/images/", S, B, img_width, img_height)
    val_dataset = CatDogDataset(dataset_splits["val"], "../data/images/", S, B, img_width, img_height)
    test_dataset = CatDogDataset(dataset_splits["test"], "../data/images/", S, B, img_width, img_height)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader)

def intersection_over_union(bbox_pred, bbox_target):
    bbox_pred_xmin = bbox_pred[..., 0:1] - (bbox_pred[..., 2:3] / 2)
    bbox_pred_ymin = bbox_pred[..., 1:2] - (bbox_pred[..., 3:4] / 2)
    bbox_pred_xmax = bbox_pred[..., 0:1] + (bbox_pred[..., 2:3] / 2)
    bbox_pred_ymax = bbox_pred[..., 1:2] + (bbox_pred[..., 3:4] / 2)

    bbox_target_xmin = bbox_target[..., 0:1] - (bbox_target[..., 2:3] / 2)
    bbox_target_ymin = bbox_target[..., 1:2] - (bbox_target[..., 3:4] / 2)
    bbox_target_xmax = bbox_target[..., 0:1] + (bbox_target[..., 2:3] / 2)
    bbox_target_ymax = bbox_target[..., 1:2] + (bbox_target[..., 3:4] / 2)

    bbox_inter_xmin = torch.max(bbox_pred_xmin, bbox_target_xmin)
    bbox_inter_ymin = torch.max(bbox_pred_ymin, bbox_target_ymin)
    bbox_inter_xmax = torch.min(bbox_pred_xmax, bbox_target_xmax)
    bbox_inter_ymax = torch.min(bbox_pred_ymax, bbox_target_ymax)

    width = (bbox_inter_xmax - bbox_inter_xmin).clamp(0)
    height = (bbox_inter_ymax - bbox_inter_ymin).clamp(0)

    intersection = width * height

    bbox_union_pred = torch.abs((bbox_pred_xmax - bbox_pred_xmin) * (bbox_pred_ymax - bbox_pred_ymin))
    bbox_union_target = torch.abs((bbox_target_xmax - bbox_target_xmin) * (bbox_target_ymax - bbox_target_ymin))

    union = bbox_union_pred + bbox_union_target - intersection

    iou = intersection / (union + 1e-6)

    return iou

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def run_nms(boxes_for_some_image, iou_threshold):

    if len(boxes_for_some_image) == 0:
        return []

    boxes_for_some_image = torch.tensor(boxes_for_some_image)
    keep_boxes_for_some_image = []

    classes = boxes_for_some_image[:, 0].unique()

    for class_idx in classes:
        mask = boxes_for_some_image[:, 0] == class_idx
        class_boxes = boxes_for_some_image[mask]

        conf_scores = class_boxes[:, 1]

        sort_indices = torch.argsort(conf_scores, descending=True)

        class_boxes = class_boxes[sort_indices]

        while True:
            keep_boxes_for_some_image.append(class_boxes[0])
            class_boxes = class_boxes[1:]
            saved_boxes = []

            for i in range(class_boxes.shape[0]):

                iou = intersection_over_union(keep_boxes_for_some_image[-1][2:], class_boxes[i][2:])

                if iou < iou_threshold:
                    saved_boxes.append(class_boxes[i])

            if len(saved_boxes) == 0:
                break

            class_boxes = torch.stack(saved_boxes)
            
            
    return [box.tolist() for box in keep_boxes_for_some_image]


def decode_preds(preds, S=7, B=2, conf_score_threshold=0.5, iou_threshold=0.5):
    preds = preds.detach().clone()
    preds = preds.reshape(preds.shape[0], -1, preds.shape[-1])

    preds[..., 10:] = torch.softmax(preds[..., 10:], dim=-1)
    
    for b in range(B):
        preds[..., 5*b + 4] = torch.sigmoid(preds[..., 5*b + 4])

    batch_size = preds.shape[0]
    grid_cells_n = preds.shape[1] 

    all_preds_boxes = []

    for i in range(batch_size):
        bboxes_for_this_image = []

        for j in range(grid_cells_n):
            row = j // S
            col = j % S

            class_probs = preds[i, j, 10:]
            max_class_prob, class_id = torch.max(class_probs, dim=0)
            class_id = class_id.item()

            for b in range(B):
                box_conf = preds[i, j, 5*b + 4].item()
                final_conf = box_conf * max_class_prob.item()

                if final_conf > conf_score_threshold:
                    x_local = preds[i, j, 5*b + 0].item()
                    y_local = preds[i, j, 5*b + 1].item()
                    w = preds[i, j, 5*b + 2].item()
                    h = preds[i, j, 5*b + 3].item()

                    x_global = (x_local + col) / S
                    y_global = (y_local + row) / S

                    clean_box = [class_id, final_conf, x_global, y_global, w, h]
                    bboxes_for_this_image.append(clean_box)

        bboxes_for_this_image = run_nms(bboxes_for_this_image, iou_threshold)

        for box in bboxes_for_this_image:
            final_row = [i] + box 
            all_preds_boxes.append(final_row)

    return all_preds_boxes


def decode_targets(targets, S=7, B=2):

    batch_size = targets.shape[0]
    grid_cells_n = S*S

    targets = targets.reshape(batch_size, grid_cells_n, -1)

    all_target_boxes = []

    for i in range(batch_size):

        bboxes_for_this_image = []

        for j in range(grid_cells_n):

            row = j // S
            col = j % S

            class_id = torch.argmax(targets[i, j, 10:]).item()

            conf = targets[i, j, 4].item()

            if conf > 0:
                x_local = targets[i, j, 0].item()
                y_local = targets[i, j, 1].item()
                w = targets[i, j, 2].item()
                h = targets[i, j, 3].item()

                x_global = (x_local + col) / S
                y_global = (y_local + row) / S

                clean_box = [class_id, conf, x_global, y_global, w, h]
                
                bboxes_for_this_image.append(clean_box)
        
        for box in bboxes_for_this_image:
            final_row = [i] + box
            all_target_boxes.append(final_row)

    return all_target_boxes


def compute_mAP(decoded_preds, decoded_targets, iou_threshold, num_classes):
    preds = torch.tensor(decoded_preds).view(-1, 7) if len(decoded_preds) > 0 else torch.zeros((0, 7))
    targets = torch.tensor(decoded_targets).view(-1, 7) if len(decoded_targets) > 0 else torch.zeros((0, 7))
    
    average_precisions = []

    for class_idx in range(num_classes):
        
        class_preds = preds[preds[:, 1] == class_idx]
        class_targets = targets[targets[:, 1] == class_idx]

        if len(class_targets) == 0 and len(class_preds) == 0:
            continue
            
        if len(class_preds) == 0:
            average_precisions.append(0.0)
            continue

        sort_idx = torch.argsort(class_preds[:, 2], descending=True)
        class_preds = class_preds[sort_idx]

        metrics_table = torch.zeros((len(class_preds), 2))
        
        targets_matched = {}
        for img_idx in class_targets[:, 0].unique():
            img_targets_count = len(class_targets[class_targets[:, 0] == img_idx])
            targets_matched[img_idx.item()] = torch.zeros(img_targets_count)

        for i, pred in enumerate(class_preds):
            img_idx = pred[0].item()
            
            img_targets = class_targets[class_targets[:, 0] == img_idx]
            
            if len(img_targets) == 0:
                metrics_table[i, 1] = 1 
                continue

            iou_scores = torch.zeros(len(img_targets))
            for j, target in enumerate(img_targets):
                if targets_matched[img_idx][j] == 0:
                    iou_scores[j] = intersection_over_union(pred[3:], target[3:])
                else:
                    iou_scores[j] = -1

            max_iou_idx = torch.argmax(iou_scores)
            max_iou = iou_scores[max_iou_idx]

            if max_iou >= iou_threshold:
                metrics_table[i, 0] = 1 
                targets_matched[img_idx][max_iou_idx.item()] = 1
            else:
                metrics_table[i, 1] = 1 
                
        cum_tp = torch.cumsum(metrics_table[:, 0], dim=0)
        cum_fp = torch.cumsum(metrics_table[:, 1], dim=0)

        total_targets = len(class_targets)
        
        recall = cum_tp / (total_targets + 1e-8)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)

        precision = torch.cat((torch.tensor([1.0]), precision))
        recall = torch.cat((torch.tensor([0.0]), recall))
        
        precision = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0)[0], dims=[0])

        AP = torch.trapz(y=precision, x=recall).item()
        
        average_precisions.append(AP)

    if len(average_precisions) == 0:
        return 0.0
        
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP