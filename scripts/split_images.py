import os
import shutil

def read_image_list(txt_path):
	image_files = []
	with open(txt_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			img_path = line.split()[0]
			image_files.append(img_path)
	return image_files

def copy_images(image_list, src_dir, dst_dir):
	os.makedirs(dst_dir, exist_ok=True)
	for img_rel_path in image_list:
		img_name = os.path.basename(img_rel_path)
		src_path = os.path.join(src_dir, img_name)
		dst_path = os.path.join(dst_dir, img_name)
		if os.path.exists(src_path):
			shutil.copy2(src_path, dst_path)
		else:
			print(f"Warning: {src_path} does not exist.")


# ===== 用户自定义路径设置区 =====
# 直接修改下方变量为你的实际路径
images_dir = r"H:\dataset\rice\160\images"  # 图片文件夹路径
train_txt = r"H:\dataset\rice\160\train.txt"  # train.txt 路径
test_txt = r"H:\dataset\rice\160\test.txt"    # test.txt 路径
train_dir = r"H:\dataset\rice\160\train"      # 训练图片输出文件夹
test_dir = r"H:\dataset\rice\160\test"        # 测试图片输出文件夹
# =================================

def main():
	train_images = read_image_list(train_txt)
	test_images = read_image_list(test_txt)
	copy_images(train_images, images_dir, train_dir)
	copy_images(test_images, images_dir, test_dir)

if __name__ == '__main__':
	main()
