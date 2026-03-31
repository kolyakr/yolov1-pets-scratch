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

def get_detection_model(num_classes):
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
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