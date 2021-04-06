import random
import numpy as np
from torch.utils.data.sampler import Sampler


class SameGroupSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)

        groups = dataset.groups

        items = zip(list(range(len(groups))), groups)

        item_to_group = {}
        group_to_items = {}

        for idx, group in items:
            item_to_group[idx] = group

            if group not in group_to_items:
                group_to_items[group] = [idx]
            else:
                group_to_items[group].append(idx)

        self.groups = set(groups)
        self.item_to_group = item_to_group
        self.group_to_items = group_to_items
        
    def __len__(self):
        return len(self.groups)
        
    def __iter__(self):
        for _ in range(len(self)):
            # Sample one group
            group_sample = random.sample(self.groups, 1)[0]
            
            items = self.group_to_items[group_sample]
            replace = False
            if len(items) < 2:
                replace = True

            # Sample two ids
            sample1, sample2 = np.random.choice(items, 2, replace=replace)
            
            yield sample1
            yield sample2