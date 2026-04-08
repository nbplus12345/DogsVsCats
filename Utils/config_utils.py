import argparse


def get_cnn_train_args():
    """获取训练专用参数"""
    parser = argparse.ArgumentParser(description="CNN_train_config")

    # 数据与网络
    parser.add_argument('--data_dir', type=str, default='./dataset/train', help='训练数据集的根目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (默认: 32)')

    # 超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量 (默认: 0.9)')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数 (默认: 100)')
    parser.add_argument('--patience', type=int, default=5, help='早停容忍轮数 (默认: 5)')

    # 保存路径
    parser.add_argument('--save_path', type=str, default='./weights/cat_dog_CNN_model.pth', help='模型权重保存路径')

    return parser.parse_args()


def get_resnet_train_args():
    """获取训练专用参数"""
    parser = argparse.ArgumentParser(description="ResNet_train_config")

    # 数据与网络
    parser.add_argument('--data_dir', type=str, default='./dataset/train', help='训练数据集的根目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (默认: 32)')

    # 超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量 (默认: 0.9)')
    parser.add_argument('--epochs', type=int, default=50, help='最大训练轮数 (默认: 50)')
    parser.add_argument('--patience', type=int, default=3, help='早停容忍轮数 (默认: 3)')

    # 保存路径
    parser.add_argument('--save_path', type=str, default='./weights/cat_dog_ResNet_model.pth', help='模型权重保存路径')

    return parser.parse_args()


def get_predict_args():
    """获取预测专用参数"""
    parser = argparse.ArgumentParser(description="🐱cat_dog_model_predict_config")

    # 预测所需参数
    parser.add_argument('--test_dir', type=str, default='./dataset/new_test', help='测试集图片目录')
    parser.add_argument('--model_name', type=str, choices=['cat_dog_ResNet_model', 'cat_dog_CNN_model'],
                        default='cat_dog_ResNet_model', help='选择要使用的模型架构')
    parser.add_argument('--batch_size', type=int, default=32, help='测试时的批次大小')

    return parser.parse_args()