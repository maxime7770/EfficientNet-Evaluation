import os
import json
from PIL import Image
import pandas as pd
import torch
import torchvision
from torchvision import transforms

# Load the dataset (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)


def load_json(file):
    ''' load a .json file and outputs a pandas Dataframe
    '''
    with open(file) as file:
        df = json.load(file)
        df = pd.DataFrame.from_dict(df, orient="index")
    return df



# to build a dataloader, we have to create a Dataset class:

class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = load_json(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.index.values[idx])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 0]
        if image.mode != "RGB": # some images have the mode 'L' and have to be converted into 'RGB'
                image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



def create_dataloader(batch_size=16, shuffle=False, pin_memory=False, num_workers=4):
    ''' creates a pytorch dataloader from our dataset
    '''
    image_size = (224, 224)
    # images have to be transformed to be fed into the pytorch model
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = Dataset(annotations_file='labels.json',
                        img_dir='./dataset',
                        transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            pin_memory=pin_memory,
                                            num_workers=num_workers)
    return dataloader

