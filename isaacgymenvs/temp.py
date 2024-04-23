import torch
import numpy as np
import matplotlib.pyplot as plt

def get_rew(pred):
    # disc_logits = torch.full((5,1), pred)
    # prob = 1 / (1 + torch.exp(-disc_logits)) 
    # disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001)))
    # # print(pred, disc_r)
    # return torch.mean(disc_r)

    disc_r = torch.maximum(torch.tensor(1 - (0.25*(pred - 1)**2)), torch.tensor(0.0001))

    return disc_r

preds = np.linspace(-2, 2, 200)
rews = np.zeros_like(preds)
for idx, pred in enumerate(preds):
    rews[idx] = get_rew(pred)


print(preds)
print(rews)

plt.plot(preds, rews)
plt.show()

# disc_r *= self._disc_reward_scale