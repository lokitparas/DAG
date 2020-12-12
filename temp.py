import torch
#torch.cuda.set_device(1)
import pycuda.driver as drv
print(torch.cuda.device_count())
print(torch.cuda.current_device())
