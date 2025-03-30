import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A

# set paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")
YOLO_LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")

# create dirs
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# uniform image size
IMG_SIZE = 640

# PASCAL VOC map to YOLO format
CLASS_MAPPING = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

# Define Albumentations data augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.GaussNoise(p=0.2)
])

# 1. read all images
def load_images(image_dir):
    if not os.path.exists(image_dir):
        print(f"Dir not found: {image_dir}")
        return []
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(images) == 0:
        print("No Image Found！")
    return images

# 2. auto match filename
def find_closest_filename(image_dir, target_filename):
    files = os.listdir(image_dir)
    lower_files = {f.lower(): f for f in files}
    return lower_files.get(target_filename.lower(), None)

# 3. adjust image size and apply augmentation
def resize_images(image_path, output_path, size=IMG_SIZE):
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)

    corrected_filename = find_closest_filename(image_dir, image_name)
    if corrected_filename:
        image_path = os.path.join(image_dir, corrected_filename)
    else:
        print(f"File {image_name} not found")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't read: {image_path}")
        return

    # Resize to fixed size first
    img = cv2.resize(img, (size, size))

    # Apply augmentation
    augmented = transform(image=img)
    img = augmented['image']

    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

# 4. decode VOC to YOLO
def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_labels = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = CLASS_MAPPING[class_name]

        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = map(int, [
            bbox.find("xmin").text, bbox.find("ymin").text,
            bbox.find("xmax").text, bbox.find("ymax").text
        ])

        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_labels

# 5. process dataset
def process_dataset():
    images = load_images(IMAGE_DIR)

    for img_name in tqdm(images, desc="Processing images"):
        image_path = os.path.join(IMAGE_DIR, img_name)
        xml_path = os.path.join(ANNOTATION_DIR, img_name.replace(".jpg", ".xml").replace(".png", ".xml"))

        if not os.path.exists(xml_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Can't read image: {image_path}")
            continue
        h, w, _ = img.shape

        resized_image_path = os.path.join(OUTPUT_DIR, img_name)
        resize_images(image_path, resized_image_path)

        yolo_labels = convert_voc_to_yolo(xml_path, w, h)
        label_file = os.path.join(YOLO_LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

# 6. split dataset
def split_dataset():
    images = load_images(OUTPUT_DIR)
    print(f"Total image#: {len(images)}")

    if len(images) == 0:
        print("Fatal: Found no Image, check `OUTPUT_DIR`")
        return

    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

    print(f"train set: {len(train_imgs)}, validation Set: {len(val_imgs)}, test set: {len(test_imgs)}")

    for split, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        split_dir = os.path.join(DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        for img in img_list:
            src_img = os.path.join(OUTPUT_DIR, img)
            dst_img = os.path.join(split_dir, img)

            src_label = os.path.join(YOLO_LABELS_DIR, img.replace(".jpg", ".txt").replace(".png", ".txt"))
            dst_label = os.path.join(split_dir, img.replace(".jpg", ".txt").replace(".png", ".txt"))

            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            else:
                print(f"can't find image: {src_img}")

            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f"can't find labeled file: {src_label}")

# 7. run data preprocessing
if __name__ == "__main__":
    print("Start Data Preprocessing...")
    process_dataset()
    split_dataset()
    print("DataPreprocessing Done！")