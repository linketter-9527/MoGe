import torch

# checkpoint 路径
checkpoint_path = "moge-2-vitl-normal.pt"

# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 查看 checkpoint 的 key
print("Checkpoint keys:", checkpoint.keys())

# 获取模型参数字典
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint  # 直接就是 state_dict

# 打印每个参数的名字和形状
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")

