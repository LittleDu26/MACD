import os
import shutil

def copy_specific_files():
    src_dir = os.path.join('result', 'MACD')
    dst_dir = os.path.join('visual', 'MACD')

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        # 计算相对于源目录的相对路径
        rel_path = os.path.relpath(root, src_dir)
        
        # 目标路径
        if rel_path == '.':
            dst_path = dst_dir
        else:
            dst_path = os.path.join(dst_dir, rel_path)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        path_parts = rel_path.split(os.sep)

        # 如果当前是名字为 '0' 的文件夹
        if path_parts[-1] == '0':
            # 只复制 pop.csv
            if 'table.csv' in files:
                src_file = os.path.join(root, 'table.csv')
                dst_file = os.path.join(dst_path, 'table.csv')
                shutil.copy2(src_file, dst_file)
            
            # 清空 dirs 列表，使得 os.walk 不会继续遍历 '0' 文件夹下的子文件夹 (如 generation_* 等)
            dirs[:] = []
        else:
            # 不是 '0' 文件夹的内容，复制其他所有文件
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                shutil.copy2(src_file, dst_file)

if __name__ == '__main__':
    copy_specific_files()
    print("Files copied successfully!")
