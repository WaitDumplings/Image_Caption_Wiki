from torch.utils.data import Dataset
import base64
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from io import BytesIO

class WikipediaDataset(Dataset):
    def __init__(self, data, split,
                 captions, transforms=None):
        """
        Customize Dataset

        :param data: 64coded data
        :param split: model type (for train / val)
        :param captions: captions
        :param transforms: initialize data
        """
        self.data = data
        self.transforms = transforms
        self.split = split
        self.captions = captions
        self.cpi = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_bytes = base64.b64decode(self.data[index]["b64_bytes"])
        img = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))

        caption = self.data[index]['caption']
        caplen = torch.tensor(self.data[index]['caplen'], dtype=torch.long)

        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((index // self.cpi) * self.cpi):(((index // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions