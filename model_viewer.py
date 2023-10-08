import torch

lz = torch.load('checkpoints/2lane_400m/model.pt')

for parameter in lz.parameters():
    print(parameter.shape ,':')
    print(parameter)
    print('-------------------------------------------')
