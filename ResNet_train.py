import random
import time
from Utils.config_utils import get_resnet_train_args
from Utils.logger_utils import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

args = get_resnet_train_args()
current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(logger_name="ResNet", log_file=f"logs/ResNet_train_{current_time}.log")
logger = log_manager.get_logger()

logger.info("")
logger.info("=== Pre-flight Checklist ===")

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    torch_directml = None
    # 如果没有安装 directml，则回退到标准的 CUDA (Nvidia GPU) 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

# 数据处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=args.data_dir, transform=val_transform)

num_data = len(train_dataset)
indices = list(range(num_data))  # 生成[0,1,2,...,24999]
random.shuffle(indices)          # 打乱这个索引列表
train_size = int(0.8 * num_data)
train_dataset = Subset(train_dataset, indices[:train_size])    # 把这个列表前80%的索引分给训练集
val_dataset = Subset(val_dataset, indices[train_size:])        # 后面的给验证集

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)  # 验证集不需要打乱
logger.info(f"[INFO] Dataset loaded. Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
logger.info(f"[INFO] Dataloaders ready. Batch size: {train_loader.batch_size} | Total train steps per epoch: {len(train_loader)}")

# 加载ResNet-18模型
model = models.resnet18(weights='DEFAULT')

# 冻结底层权重，requires_grad = False：这就相当于给前面的 18 层卷积网络加上了“只读属性”。
for param in model.parameters():
    param.requires_grad = False

# 自动获取 ResNet-18 原本最后一层 (fc) 的输入神经元个数 (通常是 512)
# 当数据流到最后一层卷积层结束时，它变成了一个形状为 512 * 7 * 7 的立体方块。
# 为了把这个立体方块送进全连接层，ResNet 的倒数第二层自带了一个叫 AdaptiveAvgPool2d 的压缩机。
# 它把每一个 7 * 7 的平面直接求平均值，压缩成一个点。
# 于是，原来的立体方块被压扁成了一根包含 512 个数字的一维向量。
num_ftrs = model.fc.in_features

# 一刀切掉原来输出 1000 个类别的层，换上我们崭新的、输出为 2 的层
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"[INFO] Model initialized: 【{model.__class__.__name__}】 | Total trainable parameters: {total_params:,}")

# 配置损失函数与优化器
criterion = nn.CrossEntropyLoss()
logger.info(f"[INFO] Criterion: {criterion.__class__.__name__}")
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
logger.info(f"[INFO] Optimizer configured: SGD (lr={args.lr}, momentum={args.momentum})")

# 准备开始训练
max_epochs = args.epochs
patience = args.patience     # 容忍 3 轮不降就早停，因为微调非常容易过拟合
logger.info(f"[INFO] Early Stopping configured. Patience: {patience} epochs, max_epochs: {max_epochs}")
lowest_val_loss = float('inf')
counter = 0

logger.info("============================")
logger.info("")
logger.info("===== Training Started ====✈")

for epoch in range(max_epochs):
    logger.info("")
    logger.info(f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] Training Phase...")
    logger.debug("     Step    |  Train loss  |    Time    ")
    logger.debug("-----------------------------------------")
    epoch_start_time = time.time()
    batch_start_time = time.time()
    model.train()
    train_loss = 0.0
    train_total = 0
    batch_total_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 这里的反向传播，只会去更新我们没冻结的最后一层 (fc层)！速度极快！
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        # torch.max(..., 1) 的作用：在第 1 维度（列的方向）上寻找最大值。它会返回两个东西：
        # 最大值是多少（比如 2.1），这个我们不关心，所以用 _（下划线，Python 中的占位符）把它当垃圾扔掉。
        # 最大值所在的索引位置（比如索引 0 代表猫，索引 1 代表狗）。这个才是我们要的最终选择，存进 predicted 里。
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        batch_total_loss += loss.item()

        if batch_idx % 100 == 99:
            # 计算这 100 个批次的平均 loss
            batch_avg_loss = batch_total_loss / 100
            batch_100_time = time.time() - batch_start_time
            logger.debug(f"   {batch_idx + 1:3d}/{len(train_loader):3d}   |    {batch_avg_loss:.4f}    |   {batch_100_time:.3f}s   ")
            # 打印完后清零，为下一个 100 批次重新累计
            batch_total_loss = 0.0
            batch_start_time = time.time()

    epoch_avg_train_loss = train_loss / train_total

    # --- 🕵️ 验证阶段 ---
    logger.info("                Validation Phase...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100 * val_correct / val_total
    epoch_time = time.time() - epoch_start_time
    logger.info("========================================================")
    logger.info(f"[Epoch Summary] | Train Loss: {epoch_avg_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    logger.info(
        f"                | Val acc   : {epoch_val_acc:.2f}% | Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s")
    logger.info("========================================================")

    if epoch_val_loss < lowest_val_loss:
        lowest_val_loss = epoch_val_loss
        counter = 0
        torch.save(model.state_dict(), args.save_path)
        logger.info(f"[SAVE] New best record! Model saved!")
    else:
        counter += 1
        logger.info(f"[WARN] No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            logger.info("[STOP] Early stopping triggered! Training halted.")
            break

