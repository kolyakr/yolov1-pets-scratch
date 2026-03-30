import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as transforms 

class CatDogDataset(Dataset):
    def __init__(self, parsed_data_list, img_folder_path, S, B, img_width, img_height):
        self.S = S
        self.B = B
        self.img_width = img_width
        self.img_height = img_height
        self.img_folder_path = img_folder_path

        self.parsed_data = parsed_data_list
        self.classes = self._create_classes()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.parsed_data)

    def __getitem__(self, index):
        item = self.parsed_data[index]
        
        img = Image.open(self.img_folder_path + item["image_filename"]).convert("RGB")

        data = self.transform(img)

        original_width = float(item["width"])
        original_height = float(item["height"])
        
        new_height = data.shape[1] 
        new_width = data.shape[2]

        height_scaling = new_height / original_height
        width_scaling = new_width / original_width

        new_boxes = []
        for original_box in item["boxes"]:
            xmin = original_box[0] * width_scaling
            ymin = original_box[1] * height_scaling
            xmax = original_box[2] * width_scaling
            ymax = original_box[3] * height_scaling

            new_boxes.append([xmin, ymin, xmax, ymax])

        breed_name = " ".join(item["image_filename"].split("_")[:-1])
        labels = [breed_name] * len(new_boxes)

        targets = {
            "labels": [self.classes[label] for label in labels],
            "bbox": new_boxes
        }

        targets = self._create_target_tensor(targets)

        return data, targets
        
    def _create_target_tensor(self, targets):
        out_dim = len(self.classes) + self.B * 5
        output = torch.zeros((self.S, self.S, out_dim))
    
        for bbox, label in zip(targets["bbox"], targets["labels"]):
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

            bbox_width = bbox_x_max - bbox_x_min
            bbox_height = bbox_y_max - bbox_y_min

            bbox_center_x = bbox_x_min + bbox_width / 2.0
            bbox_center_y = bbox_y_min + bbox_height / 2.0

            grid_cell_width = self.img_width / self.S
            grid_cell_height = self.img_height / self.S

            grid_x_coord = int(bbox_center_x / grid_cell_width)
            grid_y_coord = int(bbox_center_y / grid_cell_height)
            
            grid_x_coord = min(grid_x_coord, self.S - 1)
            grid_y_coord = min(grid_y_coord, self.S - 1)

            new_bbox_width = bbox_width / self.img_width
            new_bbox_height = bbox_height / self.img_height
            new_bbox_x = (bbox_center_x / grid_cell_width) - grid_x_coord
            new_bbox_y = (bbox_center_y / grid_cell_height) - grid_y_coord
            conf = 1.0

            output[grid_y_coord, grid_x_coord, 0] = new_bbox_x
            output[grid_y_coord, grid_x_coord, 1] = new_bbox_y
            output[grid_y_coord, grid_x_coord, 2] = new_bbox_width
            output[grid_y_coord, grid_x_coord, 3] = new_bbox_height
            output[grid_y_coord, grid_x_coord, 4] = conf
            
            output[grid_y_coord, grid_x_coord, self.B * 5 + label] = 1.0 

        return output
    
    def _create_classes(self):
        with open("../data/annotations/list.txt", "r") as f:
            text = f.read()

        classes = set([" ".join(sent.split(" ")[0].split("_")[:-1]) for sent in text.split("\n")])

        classes.discard('')

        sorted_classes = sorted(list(classes))

        class_to_id = {breed: index for index, breed in enumerate(sorted_classes)}

        return class_to_id