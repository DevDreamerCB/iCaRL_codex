import numpy as np
import torch

# result = []
# # test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
# test = torch.randn(size=(32,256)).detach().cpu().numpy()
# class_mean_set = np.random.randn(2,256)
# for target in test:
#     x = target - class_mean_set
#     x = np.linalg.norm(x, ord=2, axis=1)
#     x = np.argmin(x)
#     result.append(x)
# print(len(result))

import numpy as np

arr = np.array([1.0, 2.0, 3.0])
arr = arr[:100]
print(arr)

tensor_as = torch.as_tensor(arr)
tensor_tensor = torch.tensor(arr)

arr[0] = 100.0
print(tensor_as)       # tensor([100., 2., 3.])  # 共享内存，改变
print(tensor_tensor)   # tensor([1., 2., 3.])    # 复制独立，未改变

