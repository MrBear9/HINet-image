import torch

import lmdb
env = lmdb.open("./datasets/GoPro/train/sharp_crops.lmdb", readonly=True)
print(env.stat())  # 查看数据条目数量

print("GPU是否可用:", torch.cuda.is_available())
print("GPU设备个数:", torch.cuda.device_count())
print("查看gpu设备名称:", torch.cuda.get_device_name())
print("当前设备:", torch.cuda.current_device())