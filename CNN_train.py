from Utils.logger_utils import Logger
from Utils.config_utils import get_cnn_train_args
import random
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim


# 定义一个继承自 nn.Module 的类
class CatDogNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 特征提取部分 ---
        # 卷积 -> ReLU -> 池化
        # 图像大小从 224 -> 112 -> 56 -> 28
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        # --- 特征提取部分 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平
            nn.Linear(64 * 28 * 28, 128),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5),  # 有 50% 的概率断掉连接
            nn.Linear(128, 2)  # 输出层
        )

    def forward(self, x):
        # 数据流转路径
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    args = get_cnn_train_args()
    current_time = time.strftime("%Y%m%d_%H%M")
    log_manager = Logger(logger_name="CatDogNet", log_file=f"logs/CNN_train_{current_time}.log")
    logger = log_manager.get_logger()
    logger.info("")
    logger.info("=== Pre-flight Checklist ===")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        torch_directml = None
        # 如果没有安装 directml，则回退到标准的 CUDA (Nvidia GPU) 或 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Device set to: {device}")

    train_transform = transforms.Compose([
        # 把所有图片强制缩放成 224 x 224 的大小
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率左右翻转
        transforms.RandomRotation(15),  # 随机旋转 -15度 到 15度
        # 把图片转换成神经网络能消化的 Tensor 张量
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=args.data_dir, transform=val_transform)

    # 抽样生成验证集
    num_data = len(train_dataset)
    indices = list(range(num_data))  # 生成[0,1,2,...,24999]
    random.shuffle(indices)  # 打乱这个索引列表
    train_size = int(0.8 * num_data)
    train_dataset = Subset(train_dataset, indices[:train_size])  # 把这个列表前80%的索引分给训练集
    val_dataset = Subset(val_dataset, indices[train_size:])  # 后面的给验证集

    # 创建两个传送带
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)  # 验证集不需要打乱
    logger.info(f"[INFO] Dataset loaded. Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    logger.info(f"[INFO] Dataloaders ready. Batch size: {train_loader.batch_size} | Total train steps per epoch: {len(train_loader)}")
    # 实例化
    model = CatDogNet()
    model.to(device)
    # 遍历所有参数(p)，如果需要求导(requires_grad)，就把它的元素个数(numel)加起来
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 前面是利用反射自动获取模型实例的类名，后面的:,自动加千分位 (比如 1245600 会变成 1,245,600)
    logger.info(f"[INFO] Model initialized: 【{model.__class__.__name__}】 | Total trainable parameters: {total_params:,}")

    # 实例化损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    logger.info(f"[INFO] Criterion: {criterion.__class__.__name__}")

    # 实例化优化器：最经典的 SGD 优化器
    # momentum=0.9 (动量)：这是一个极其重要的超参数，它能利用物理学中“惯性”的数学原理，
    # 帮助基础的 SGD 算法加速冲过局部的坑洼，弥补它在收敛速度上不如 Adam 的劣势。
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    logger.info(f"[INFO] Optimizer configured: SGD (lr={args.lr}, momentum={args.momentum})")

    # 早停参数
    lowest_val_loss = float('inf')  # 初始设为无穷大
    patience = args.patience  # 连续 5 轮不进步就停
    max_epochs = args.epochs  # 给一个足够大的上限，反正会自动停
    logger.info(f"[INFO] Early Stopping configured. Patience: {patience} epochs, max_epochs: {max_epochs}")
    counter = 0  # 计数器
    logger.info("============================")
    logger.info("")
    logger.info("===== Training Started ====✈")

    for epoch in range(max_epochs):
        model.train()  # 告诉模型：现在是学习状态
        logger.info("")
        logger.info(f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] Training Phase...")
        logger.debug("     Step    |  Train loss  |    Time    ")
        logger.debug("-----------------------------------------")
        epoch_start_time = time.time()
        batch_start_time = time.time()
        train_total = 0
        batch_total_loss = 0.0
        epoch_total_loss = 0.0
        # enumerate 会自动从 train_loader 中按批次提取数据
        # batch_idx 是当前批次的索引，(inputs, labels) 是具体的数据张量
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 【极其重要】如果使用了 GPU/NPU 加速，必须将数据也转移到相同设备上
            # 否则模型在 GPU，数据在 CPU，矩阵乘法会直接报错
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 标准的“五步法”数学闭环
            # 1. 梯度清零 (Zero Gradients)
            optimizer.zero_grad()
            # 2. 前向传播 (Forward Pass)
            outputs = model(inputs)
            # 3. 计算损失 (Calculate Loss)
            loss = criterion(outputs, labels)
            # 4. 反向传播求导 (Backward Pass)
            loss.backward()
            # 5. 更新权重参数 (Optimizer Step)
            optimizer.step()

            # 开始记录平均损失
            # loss.item() 将包含单个数的 PyTorch 张量转换损失值本身
            batch_total_loss += loss.item()
            epoch_total_loss += loss.item()
            # 开始记录预测准确率
            train_predicted = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)

            # 每处理 100 个批次，在终端打印一次当前的平均误差
            if batch_idx % 100 == 99:
                batch_100_time = time.time() - batch_start_time
                # 计算这 100 个批次的平均 loss
                batch_avg_loss = batch_total_loss / 100
                logger.debug(f"   {batch_idx + 1:3d}/{len(train_loader):3d}   |    {batch_avg_loss:.4f}    |   {batch_100_time:.3f}s   ")
                # 打印完后清零，为下一个 100 批次重新累计
                batch_total_loss = 0.0
                batch_start_time = time.time()
        epoch_avg_train_loss = epoch_total_loss / len(train_loader)
        logger.info("                Validation Phase...")
        model.eval()  # 告诉模型：现在是考试状态
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 考试时不需要算梯度，省内存
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算准确率
                val_predicted = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        # 打印当前 Epoch 的综合成绩单
        logger.info("========================================================")
        logger.info(f"[Epoch Summary] | Train Loss: {epoch_avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"                | Val acc   : {val_accuracy:.2f}% | Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s")
        logger.info("========================================================")
        # 早停机制
        if avg_val_loss < lowest_val_loss:
            # 情况 A：模型进步了！
            lowest_val_loss = avg_val_loss
            counter = 0  # 重置耐心计数器
            torch.save(model.state_dict(), args.save_path)
            logger.info(f"[SAVE] New best record! Model saved!")

        else:
            # 情况 B：模型没进步（甚至反弹了）
            counter += 1
            logger.info(f"[WARN] No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                logger.info("[STOP] Early stopping triggered! Training halted.")
                break  # 跳出大循环，提前结束训练
