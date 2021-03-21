import torch
from torch import Tensor

def cosine_similarity(vec1: Tensor, vec2: Tensor):
    if not isinstance(vec1, Tensor):
        vec1 = torch.tensor(vec1)

    if not isinstance(vec2, Tensor):
        vec2 = torch.tensor(vec2)

    not_list = False

    if len(vec1.shape) == 1 and len(vec2.shape) == 1:
        not_list = True

    if len(vec1.shape) == 1:
        vec1 = vec1.unsqueeze(0)

    if len(vec2.shape) == 1:
        vec2 = vec2.unsqueeze(0)

    sim = torch.cosine_similarity(vec1, vec2).tolist()

    if not_list:
        sim = sim[0]

    return sim