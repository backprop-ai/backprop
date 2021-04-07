import torch
from torch.utils.data import Dataset
from PIL import Image

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
    

class TextToTextDataset(Dataset):
    def __init__(self, inputs, outputs, process_text, max_input_length, max_output_length):
        super().__init__()
        
        self.inputs = inputs
        self.outputs = outputs
        self.process_text = process_text
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        outputs = self.outputs[idx]

        inputs, outputs = self.process_text(inputs, outputs, self.max_input_length, self.max_output_length)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.squeeze(0)
            outputs = outputs.squeeze(0)
        else:
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            outputs = {k: v.squeeze(0) for k, v in outputs.items()}


        return {**inputs, **outputs}
