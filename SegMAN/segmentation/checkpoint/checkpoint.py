import torch

# 加载 checkpoint
checkpoint_path = "segman_l_cityscapes.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 查看 checkpoint 内的 key
print(checkpoint.keys())  # 一般会有 'state_dict' 或 'model'

# 获取模型参数字典
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint  # 有些 checkpoint 就直接是 state_dict

# 打印每个参数的名字和大小
for name, param in state_dict.items():
    print(name, param.shape)


