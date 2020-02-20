import torch

#a = torch.randn((3, 3), requires_grad=True)
#w1 = torch.randn((3, 3), requires_grad=True)

a = torch.randn((3, 3), requires_grad=True)
w1 = torch.randn((3, 3), requires_grad=True)
w2 = torch.randn((3, 3), requires_grad=True)
w3 = torch.randn((3, 3), requires_grad=True)

b = w1 * a
c = w2 * b

L = w3 * c
