import re


def filter_annotations(input_file, output_file):
    """
    读取标注文件，过滤掉：
    1. BLAST2相关的行
    2. BLAST1中从001到080的图片
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        # 检查是否包含BLAST2，如果包含则跳过
        if 'BLAST2' in line:
            continue
        
        # 检查是否是BLAST1的001-080
        # 使用正则表达式匹配 BLAST1_XXX 格式
        match = re.search(r'BLAST1_(\d+)', line)
        if match:
            number = int(match.group(1))
            # 如果编号在1到80之间，跳过
            if 1 <= number <= 80:
                continue
        
        # 保留这一行
        filtered_lines.append(line)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    print(f"处理完成！")
    print(f"原始行数: {len(lines)}")
    print(f"过滤后行数: {len(filtered_lines)}")
    print(f"移除行数: {len(lines) - len(filtered_lines)}")


if __name__ == '__main__':
    # 设置输入和输出文件路径
    input_file = 'H:\\dataset\\rice\\200\\all_list.txt'  # 修改为你的输入文件路径
    output_file = 'H:\\dataset\\rice\\200\\test80.txt'  # 修改为你的输出文件路径

    filter_annotations(input_file, output_file)
