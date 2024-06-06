import torch

p0 = torch.zeros((100,1))
p1 = torch.ones((100,1))

obs = torch.full((100,210), 1.23)

print(p0.shape)
print(p1.shape)
print(obs.shape)

s1, s0 = torch.chunk(obs, chunks=2, dim=-1)

print(s1)
print(s0)

print(s1.shape)
print(s0.shape)

s1 = torch.cat((p1,s1), -1)
s0 = torch.cat((p0,s0), -1)

print(s1.shape)
print(s0.shape)

print(s1)
print(s0)

final = torch.cat((s1,s0), -1)

print(final.shape)
print(final)
print(final[0])