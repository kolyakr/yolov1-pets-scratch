import os
import xml.etree.ElementTree as ET

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