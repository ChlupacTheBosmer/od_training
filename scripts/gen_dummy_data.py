import os
from PIL import Image
from pathlib import Path
import yaml

def gen_data():
    # Dirs
    yolo_structure = ['data/dummy_yolo/images', 'data/dummy_yolo/labels']
    coco_structure = ['data/dummy_coco/train', 'data/dummy_coco/valid', 'data/dummy_coco/test']
    
    for d in yolo_structure + coco_structure:
        os.makedirs(d, exist_ok=True)
        
    # Image
    img = Image.new('RGB', (640, 640), color = 'red')
    img.save('data/dummy_yolo/images/test.jpg')
    
    # Label (YOLO)
    with open('data/dummy_yolo/labels/test.txt', 'w') as f:
        f.write("0 0.5 0.5 0.2 0.2")
        
    # YAML
    data_yaml = {
        'path': os.path.abspath('data/dummy_yolo'),
        'train': 'images',
        'val': 'images',
        'names': {0: 'object'}
    }
    with open('data/dummy_yolo/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
        
    print("YOLO Dummy Data Created.")

if __name__ == "__main__":
    gen_data()
