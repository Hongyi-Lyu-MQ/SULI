import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
FORGET_CLASS_IDXS = [1]  # 例如 class 1（car）
DATA_DIR = './data'
