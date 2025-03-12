import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A

# 设置路径
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")
YOLO_LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")

# 创建目录
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 统一图片大小
IMG_SIZE = 640

# PASCAL VOC 类别映射到 YOLO 格式
CLASS_MAPPING = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

# 1️⃣ 读取所有图片（自动匹配大小写）
def load_images(image_dir):
    if not os.path.exists(image_dir):
        print(f"❌ 目录不存在: {image_dir}")
        return []
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(images) == 0:
        print("❌ 没有找到任何图片，请检查路径和文件格式！")
    return images

# 2️⃣ 自动匹配最接近的文件名（防止拼写或大小写问题）
def find_closest_filename(image_dir, target_filename):
    """ 在目录中找到最接近的文件名（忽略大小写和拼写错误） """
    files = os.listdir(image_dir)
    lower_files = {f.lower(): f for f in files}  # 生成小写文件名的映射
    return lower_files.get(target_filename.lower(), None)

# 3️⃣ 调整图片大小（支持自动匹配文件名）
def resize_images(image_path, output_path, size=IMG_SIZE):
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)

    # 自动匹配最接近的文件名
    corrected_filename = find_closest_filename(image_dir, image_name)
    if corrected_filename:
        image_path = os.path.join(image_dir, corrected_filename)
    else:
        print(f"❌ 文件 {image_name} 不存在")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    img = cv2.resize(img, (size, size))
    cv2.imwrite(output_path, img)
    print(f"✅ 已保存: {output_path}")

# 4️⃣ 解析 PASCAL VOC XML 并转换为 YOLO 格式
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

        # 归一化边界框
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_labels

# 5️⃣ 处理所有数据
def process_dataset():
    images = load_images(IMAGE_DIR)

    for img_name in tqdm(images, desc="Processing images"):
        image_path = os.path.join(IMAGE_DIR, img_name)
        xml_path = os.path.join(ANNOTATION_DIR, img_name.replace(".jpg", ".xml").replace(".png", ".xml"))

        if not os.path.exists(xml_path):
            continue  # 如果没有对应的 XML 标注，跳过

        # 读取原始图片大小
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 无法读取图片: {image_path}")
            continue
        h, w, _ = img.shape

        # 处理图片
        resized_image_path = os.path.join(OUTPUT_DIR, img_name)
        resize_images(image_path, resized_image_path)

        # 转换标签
        yolo_labels = convert_voc_to_yolo(xml_path, w, h)
        label_file = os.path.join(YOLO_LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

# 6️⃣ 划分数据集
def split_dataset():
    images = load_images(OUTPUT_DIR)
    print(f"🔍 总图片数: {len(images)}")

    if len(images) == 0:
        print("❌ 错误：没有找到任何图片，请检查 `OUTPUT_DIR`")
        return
    
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

    print(f"✅ 训练集: {len(train_imgs)}, 验证集: {len(val_imgs)}, 测试集: {len(test_imgs)}")

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
                print(f"⚠️ 找不到图片: {src_img}")

            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f"⚠️ 找不到标签文件: {src_label}")

# 7️⃣ 运行数据预处理
if __name__ == "__main__":
    print("🚀 开始数据预处理...")
    process_dataset()   # 处理图片 & 标签
    split_dataset()     # 划分数据集
    print("✅ 数据预处理完成！")
