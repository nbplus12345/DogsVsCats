import os
import random
import shutil

def create_holdout_test_set(train_dir='../dataset/train', test_dir='../dataset/new_test', num_per_class=500):
    categories = ['cat', 'dog']

    # 1. 创建测试集目录结构
    for cat in categories:
        target_path = os.path.join(test_dir, cat)
        os.makedirs(target_path, exist_ok=True)

    print(f"开始从 {train_dir} 抽取数据到 {test_dir}...")

    # 2. 对每个类别进行精确抽样和移动
    for cat in categories:
        src_path = os.path.join(train_dir, cat)
        dst_path = os.path.join(test_dir, cat)

        # 获取该类别下所有的图片文件名
        all_images = os.listdir(src_path)

        # 随机抽取 500 张 (保证每次抽样的随机性，也可以加 random.seed() 固定)
        random.seed(42)
        test_images = random.sample(all_images, num_per_class)

        # 将抽中的图片移动（剪切）到 test 文件夹
        for img_name in test_images:
            src_img_path = os.path.join(src_path, img_name)
            dst_img_path = os.path.join(dst_path, img_name)
            shutil.move(src_img_path, dst_img_path)  # 注意这里是 move (剪切)

        print(f"成功将 {num_per_class} 张 {cat} 图片转移至测试集！")

    print("\n✅ 测试集物理隔离完成！")
    print(f"现在 train 目录下还剩 24000 张图，可用于接下来的 80/20 训练与验证。")


if __name__ == '__main__':
    create_holdout_test_set()