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

PATH = "datasets/Flicker8k_Dataset/"



bpe = yttm.BPE(model = 'bpe.model')


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, path, images, all_data, max_len):
        self.images = images
        self.data = all_data
        self.path = path
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if img.startswith("datasets/Flicker8k_Dataset/") :
            img = img[27:]
        sen = self.data[img]
        sen = sen[random.randint(0, 4)]
        img = T.functional.resize(torchvision.io.read_image(PATH + img) / 255, (224, 224), antialias = True)
        src = bpe.encode(sen, bos = True, eos = False)
        tgt = bpe.encode(sen, bos = False, eos = True)
        src = src + [0] * (max_len - len(src))
        src = torch.LongTensor(src)
        tgt = tgt + [0] * (max_len - len(tgt))
        tgt = torch.LongTensor(tgt)
        return img, src, tgt