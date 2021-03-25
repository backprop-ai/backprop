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

        texts1 = self.tokenize_text(self.texts1[idx]).squeeze(0)
        texts2 = self.tokenize_text(self.texts2[idx]).squeeze(0)
        
        imgs1 = self.transform_img(Image.open(self.imgs1[idx])).squeeze(0)
        imgs2 = self.transform_img(Image.open(self.imgs2[idx])).squeeze(0)

        similarity_scores = torch.tensor(self.similarity_scores[idx])

        return texts1, imgs1, texts2, imgs2, similarity_scores