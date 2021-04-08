import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageTextPairDataset(Dataset):
    def __init__(self, img_text_pairs1, img_text_pairs2, similarity_scores,
                transform_img, tokenize_text):
        super().__init__()
        self.texts1 = [t1 for i1, t1 in img_text_pairs1]
        self.texts2 = [t2 for i2, t2 in img_text_pairs2]

        self.imgs1 = [i1 for i1, t1 in img_text_pairs1]
        self.imgs2 = [i2 for i2, t2 in img_text_pairs2]

        self.similarity_scores = similarity_scores

        self.transform_img = transform_img
        self.tokenize_text = tokenize_text

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):
        texts1 = self.tokenize_text(self.texts1[idx])
        texts2 = self.tokenize_text(self.texts2[idx])

        if isinstance(texts1, torch.Tensor):
            texts1 = texts1.squeeze(0)
        else:
            texts1 = {k: v.squeeze(0) for k, v in texts1.items()}

        if isinstance(texts2, torch.Tensor):
            texts2 = texts2.squeeze(0)
        else:
            texts2 = {k: v.squeeze(0) for k, v in texts2.items()}

        imgs1 = self.transform_img(Image.open(self.imgs1[idx])).squeeze(0)
        imgs2 = self.transform_img(Image.open(self.imgs2[idx])).squeeze(0)

        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return texts1, imgs1, texts2, imgs2, similarity_scores

class ImagePairDataset(Dataset):
    def __init__(self, imgs1, imgs2, similarity_scores,
                transform_img):
        super().__init__()

        self.imgs1 = imgs1
        self.imgs2 = imgs2

        self.similarity_scores = similarity_scores

        self.transform_img = transform_img

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):

        imgs1 = self.transform_img(Image.open(self.imgs1[idx])).squeeze(0)
        imgs2 = self.transform_img(Image.open(self.imgs2[idx])).squeeze(0)

        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return imgs1, imgs2, similarity_scores

class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, similarity_scores,
                tokenize_text):
        super().__init__()

        self.texts1 = texts1
        self.texts2 = texts2

        self.similarity_scores = similarity_scores

        self.tokenize_text = tokenize_text

    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):
        texts1 = self.tokenize_text(self.texts1[idx])
        texts2 = self.tokenize_text(self.texts2[idx])

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
    def __init__(self, images, texts, groups, transform_img, tokenize_text):
        super().__init__()

        self.images = images
        self.texts = texts
        self.groups = groups

        self.transform_img = transform_img
        self.tokenize_text = tokenize_text

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform_img(Image.open(self.images[idx])).squeeze(0)
        text = self.tokenize_text(self.texts[idx])

        if isinstance(text, torch.Tensor):
            text = text.squeeze(0)
        else:
            text = {k: v.squeeze(0) for k, v in text.items()}

        group = torch.tensor(self.groups[idx])

        return image, text, group

class ImageGroupDataset(Dataset):
    def __init__(self, images, groups, transform_img):
        super().__init__()

        self.images = images
        self.groups = groups

        self.transform_img = transform_img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform_img(Image.open(self.images[idx])).squeeze(0)

        group = torch.tensor(self.groups[idx])

        return image, group

class TextGroupDataset(Dataset):
    def __init__(self, texts, groups, tokenize_text):
        super().__init__()

        self.texts = texts
        self.groups = groups

        self.tokenize_text = tokenize_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.tokenize_text(self.texts[idx])

        if isinstance(text, torch.Tensor):
            text = text.squeeze(0)
        else:
            text = {k: v.squeeze(0) for k, v in text.items()}

        group = torch.tensor(self.groups[idx])

        return text, group

class SingleLabelImageClassificationDataset(Dataset):
    def __init__(self, images, labels, process_image):
        super().__init__()

        self.images = images
        self.labels = labels
        self.label_to_idx = {label: i for i, label in enumerate(set(labels))}

        self.process_image = process_image
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.process_image(image).squeeze(0)

        target = torch.tensor(self.label_to_idx[self.labels[idx]])

        return image, target


class MultiLabelImageClassificationDataset(Dataset):
    def __init__(self, images, labels, process_image):
        super().__init__()

        self.images = images
        self.labels = labels
        all_labels = list(np.concatenate(labels).flat)
        self.all_labels = set(all_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        self.process_image = process_image
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.process_image(image).squeeze(0)

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
        params = {k: (v if type(v) != list else v[idx]) for k,v in self.params.items()}

        inp = self.process_batch(params, task=self.task)

        if isinstance(inp, torch.Tensor):
            inp = inp.squeeze(0)
            # out = out.squeeze(0)
        else:
            inp = {k: v.squeeze(0) for k, v in inp.items()}
            # out = {k: v.squeeze(0) for k, v in out.items()}

        return {**inp}

class SingleLabelTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, process_text, max_input_length):
        super().__init__()

        self.texts = texts
        self.labels = labels
        self.label_to_idx = {label: i for i, label in enumerate(set(labels))}
        self.max_input_length = max_input_length
        self.process_text = process_text
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        target = self.label_to_idx[self.labels[idx]]
        inp = self.process_text(self.texts[idx], target, self.max_input_length)

        return inp