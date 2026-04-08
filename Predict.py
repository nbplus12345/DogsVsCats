import sys
import time
from torch import nn
from torch.utils.data import DataLoader
from CNN_train import CatDogNet
import torch
from Utils.config_utils import get_predict_args
from Utils.logger_utils import Logger
from torchvision import transforms, datasets, models


args = get_predict_args()
test_dir = args.test_dir
model_name = args.model_name
weight_path = f'./weights/{model_name}.pth'

current_time = time.strftime("%Y%m%d_%H%M")
log_manager = Logger(logger_name="ResNet", log_file=f"{model_name}_test_{current_time}.log")
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

# 1、加载网络
if model_name == 'cat_dog_ResNet_model':
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
elif model_name == 'cat_dog_CNN_model':
    model = CatDogNet().to(device)
else:
    logger.warn("[ERROR] Invalid model weight configuration")
    sys.exit(0)
logger.info(f"[INFO] Model weight path is {weight_path}")

# 2、加载模型权重
model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))

# 3、开启评估模式
model.eval()

# 4、处理图片
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)    # shuffle=False，按顺序批改即可
logger.info(f"[INFO] Dataset loaded. Test: {len(test_dataset)} images")
logger.info(f"[INFO] Dataloaders ready. Batch size: {test_loader.batch_size} | Total test steps: {len(test_loader)}")
logger.info("============================")

correct = 0
total = 0

logger.info("")
logger.info("==== Evaluation Started ===✈")
# torch.no_grad() 告诉底层：现在是推理，不要算偏导数了
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 模型推理
        outputs = model(inputs)

        # 找到概率最大的那个选项 (0 或 1)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# 计算最终得分
accuracy = 100 * correct / total

logger.info("")
logger.info("== Evaluation Summary ==")
logger.info(f"Final Accuracy: {accuracy:.2f}%")
logger.info("========================")
