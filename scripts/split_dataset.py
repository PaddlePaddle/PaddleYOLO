import random

def split_dataset(input_txt, train_txt, test_txt, train_ratio=0.8, seed=42):
	# 读取所有行
	with open(input_txt, 'r', encoding='utf-8') as f:
		lines = [line.strip() for line in f if line.strip()]

	# 打乱顺序
	random.seed(seed)
	random.shuffle(lines)

	# 计算分割点
	split_idx = int(len(lines) * train_ratio)
	train_lines = lines[:split_idx]
	test_lines = lines[split_idx:]

	# 写入train.txt
	with open(train_txt, 'w', encoding='utf-8') as f:
		for line in train_lines:
			f.write(line + '\n')

	# 写入test.txt
	with open(test_txt, 'w', encoding='utf-8') as f:
		for line in test_lines:
			f.write(line + '\n')

if __name__ == "__main__":
	# 修改为你的输入文件路径
	input_txt = "H:\\dataset\\rice\\rice_plus_neck\\all_list.txt"  # 源txt文件名
	train_txt = "H:\\dataset\\rice\\rice_plus_neck\\train.txt"
	test_txt = "H:\\dataset\\rice\\rice_plus_neck\\test.txt"
	split_dataset(input_txt, train_txt, test_txt)
