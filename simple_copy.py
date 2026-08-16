import os
import shutil


# 与 simple_gif.py 相同的风格：先定义 root_dir 方便拼路径
root_dir = os.path.dirname(os.path.abspath(__file__))


def copy_robot_assets(env_name, id,num):

    src_pt = os.path.join(root_dir, 'result', f'{env_name}', f'{num}','controllers', f'{id}.pt')
    src_npz = os.path.join(root_dir, 'result', f'{env_name}', f'{num}','structures', f'{id}.npz')

    # 目标目录与目标路径
    dst_dir = os.path.join(root_dir, 'visual', env_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst_pt = os.path.join(dst_dir, f'{id}.pt')
    dst_npz = os.path.join(dst_dir, f'{id}.npz')

    # 校验源文件是否存在
    missing = []
    if not os.path.isfile(src_pt):
        missing.append(src_pt)
    if not os.path.isfile(src_npz):
        missing.append(src_npz)
    if missing:
        raise FileNotFoundError('以下源文件不存在：\n' + "\n".join(missing))

    # 复制
    shutil.copy2(src_pt, dst_pt)
    print(f'Copied: {src_pt} -> {dst_pt}')
    shutil.copy2(src_npz, dst_npz)
    print(f'Copied: {src_npz} -> {dst_npz}')
    print('Done.')


if __name__ == '__main__':
    env_name = 'ObstacleTraverser-v0'
    num=0
    id_list = [1208]
    for id in id_list:
        copy_robot_assets(env_name, id,num)