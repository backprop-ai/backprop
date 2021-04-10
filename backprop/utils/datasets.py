import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageTextPairDataset(Dataset):
    def __init__(self, img_text_pairs1, img_text_pairs2, similarity_scores, process_batch):
        super().__init__()
        self.texts1 = [t1 for i1, t1 in img_text_pairs1]
        self.texts2 = [t2 for i2, t2 in img_text_pairs2]

        self.imgs1 = [i1 for i1, t1 in img_text_pairs1]
        self.imgs2 = [i2 for i2, t2 in img_text_pairs2]

        self.similarity_scores = similarity_scores

        self.process_batch = process_batch

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):
        texts1 = self.process_batch({"text": self.texts1[idx]}, task="text-vectorisation")
        texts2 = self.process_batch({"text": self.texts2[idx]}, task="text-vectorisation")

        if isinstance(texts1, torch.Tensor):
            texts1 = texts1.squeeze(0)
        else:
            texts1 = {k: v.squeeze(0) for k, v in texts1.items()}

        if isinstance(texts2, torch.Tensor):
            texts2 = texts2.squeeze(0)
        else:
            texts2 = {k: v.squeeze(0) for k, v in texts2.items()}

        imgs1 = self.process_batch({"image": self.imgs1[idx]}, task="image-vectorisation")
        imgs2 = self.process_batch({"image": self.imgs2[idx]}, task="image-vectorisation")

        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return texts1, imgs1, texts2, imgs2, similarity_scores

class ImagePairDataset(Dataset):
    def __init__(self, imgs1, imgs2, similarity_scores,
                process_batch):
        super().__init__()

        self.imgs1 = imgs1
        self.imgs2 = imgs2

        self.similarity_scores = similarity_scores

        self.process_batch = process_batch

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):

        imgs1 = self.process_batch({"image": self.imgs1[idx]}, task="image-vectorisation")
        imgs2 = self.process_batch({"image": self.imgs2[idx]}, task="image-vectorisation")

        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return imgs1, imgs2, similarity_scores

class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, similarity_scores,
                process_batch):
        super().__init__()

        self.texts1 = texts1
        self.texts2 = texts2

        self.similarity_scores = similarity_scores

        self.process_batch = process_batch

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):
        texts1 = self.process_batch({"text": self.texts1[idx]}, task="text-vectorisation")
        texts2 = self.process_batch({"text": self.texts2[idx]}, task="text-vectorisation")

        if isinstance(texts1, torch.Tensor):
            texts1 = texts1.squeeze(0)
        else:
            texts1 = {k: v.squeeze(0) for k, v in texts1.items()}

        if isinstance(texts2, torch.Tensor):
            texts2 = texts2.squeeze(0)
        else:
            texts2 = {k: v.squeeze(0) for k, v in texts2.items()}
        
        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return texts1, texts2, similarity_scores

class ImageTextGroupDataset(Dataset):
    def __init__(self, images, texts, groups, process_batch):
        super().__init__()

        self.images = images
        self.texts = texts
        self.groups = groups

        self.process_batch = process_batch

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.process_batch({"image": self.images[idx]}, task="image-vectorisation")
        text = self.process_batch({"text": self.texts[idx]}, task="text-vectorisation")

        if isinstance(text, torch.Tensor):
            text = text.squeeze(0)
        else:
            text = {k: v.squeeze(0) for k, v in text.items()}

        group = torch.tensor(self.groups[idx])

        return image, text, group

class ImageGroupDataset(Dataset):
    def __init__(self, images, groups, process_batch):
        super().__init__()

        self.images = images
        self.groups = groups

        self.process_batch = process_batch

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.process_batch({"image": self.images[idx]}, task="image-vectorisation")

        group = torch.tensor(self.groups[idx])

        return image, group

class TextGroupDataset(Dataset):
    def __init__(self, texts, groups, process_batch):
        super().__init__()

        self.texts = texts
        self.groups = groups

        self.process_batch = process_batch

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.process_batch({"text": self.texts[idx]})

        if isinstance(text, torch.Tensor):
            text = text.squeeze(0)
        else:
            text = {k: v.squeeze(0) for k, v in text.items()}

        group = torch.tensor(self.groups[idx])
        return text, group

class SingleLabelImageClassificationDataset(Dataset):
    def __init__(self, images, labels, process_batch):
        super().__init__()

        self.images = images
        self.labels = labels
        self.label_to_idx = {label: i for i, label in enumerate(set(labels))}

        self.process_batch = process_batch
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.process_batch({"image": self.images[idx]}, task="image-classification")
        target = torch.tensor(self.label_to_idx[self.labels[idx]])

        return image, target


class MultiLabelImageClassificationDataset(Dataset):
    def __init__(self, images, labels, process_batch):
        super().__init__()

        self.images = images
        self.labels = labels
        all_labels = list(np.concatenate(labels).flat)
        self.all_labels = set(all_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        self.process_batch = process_batch
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.process_batch({"image": images[idx]}, task="image-classification")
        target = torch.zeros(len(self.all_labels))

        for label in self.labels[idx]:
            i = self.label_to_idx[label]
            target[i] = 1.

        return image, target

class TextToTextDataset(Dataset):
    def __init__(self, params, task, process_batch, length):
        self.params = params
        self.task = task
        self.process_batch = process_batch
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        # self.params is a dict containig lists (inputs, outputs) and fixed values (e.g. max_input_length)
        # Line here gets [idx] of lists, as well as fixed values, as a dict to be passed to model for processing.
        params = {k: (v if type(v) != list else v[idx]) for k, v in self.params.items()}

        inp = self.process_batch(params, task=self.task)

        if isinstance(inp, torch.Tensor):
            inp = inp.squeeze(0)
            # out = out.squeeze(0)
        else:
            inp = {k: v.squeeze(0) for k, v in inp.items()}
            # out = {k: v.squeeze(0) for k, v in out.items()}

        return {**inp}

class SingleLabelTextClassificationDataset(Dataset):
    def __init__(self, params, proces_batch, length):
        super().__init__()

        self.params = params
        self.process_batch = process_batch
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # self.params is a dict containig lists (inputs, outputs) and fixed values (e.g. max_input_length)
        # Line here gets [idx] of lists, as well as fixed values, as a dict to be passed to model for processing.
        params = {k: (v if type(v) != list else v[idx]) for k, v in self.params.items()}
        inp = self.process_batch(params, task="text-classification")

        return {**inp}