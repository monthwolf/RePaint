import os
import argparse
from PIL import Image, ImageDraw
import random

def create_mask(image_path, output_dir, mask_type='box'):
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size

        # 白色背景 (255) = 保留原图区域
        # 黑色 (0) = 需要修复(Inpaint)的区域
        mask = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(mask)

        if mask_type == 'box':
            # 在中心创建一个黑色矩形框 (大约占图像的50%)
            box_w, box_h = w // 2, h // 2
            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            draw.rectangle((x1, y1, x1 + box_w, y1 + box_h), fill=0)

        elif mask_type == 'random':
            # 随机线条遮罩
            for _ in range(5):
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                x2 = random.randint(0, w)
                y2 = random.randint(0, h)
                width = random.randint(w // 20, w // 10)
                draw.line((x1, y1, x2, y2), fill=0, width=width)

        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)
        mask.save(save_path)
        print(f"Generated mask for {filename} -> {save_path}")

    except Exception as e:
        print(f"Skipping {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="生成Inpainting所需的Mask掩码图像")
    parser.add_argument('--image_dir', type=str, required=True, help='包含源图像的文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='保存生成的Mask的文件夹路径')
    parser.add_argument('--type', type=str, default='box', choices=['box', 'random'], help='Mask类型: "box"(矩形) 或 "random"(随机线条)')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 支持的图片扩展名
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    # 筛选出符合扩展名的文件
    files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Found {len(files)} images. Generating masks...")
    # 遍历所有图片并生成对应的Mask
    for f in files:
        create_mask(os.path.join(args.image_dir, f), args.output_dir, args.type)

    print("Done.")

if __name__ == '__main__':
    main()
