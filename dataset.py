import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import youtokentome as yttm
import random
import yaml 

config = yaml.load(open("config.yaml", "r"), Loader = yaml.FullLoader)


max_len = config["MAX_LEN"]
vocab_size = config["VOCAB_SIZE"]
height = config['HEIGHT']
width = config['WIDTH']

PATH = "datasets/Flicker8k_Dataset/"


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, path, images, all_data, max_len, bpe_model):
        self.images = images
        self.data = all_data
        self.path = path
        self.max_len = max_len
        self.bpe = yttm.BPE(model = bpe_model)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        sen = self.data[img]
        sen = sen[random.randint(0, 4)]
        img = T.functional.resize(torchvision.io.read_image(PATH + img) / 255, (height, width), antialias = True)
        src = self.bpe.encode(sen, bos = True, eos = False)
        tgt = self.bpe.encode(sen, bos = False, eos = True)
        src = src + [0] * (self.max_len - len(src)) #Padding upto max_len
        src = torch.LongTensor(src)
        tgt = tgt + [0] * (self.max_len - len(tgt)) #Padding upto max_len
        tgt = torch.LongTensor(tgt)
        return img, src, tgt