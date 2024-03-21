import torch
import os
from tqdm import tqdm
import lightning as L
import youtokentome as yttm
from model import Image2TextTransformer
import yaml
from dataset import ImageCaptionDataset
config = yaml.load(open("config.yaml", "r"), Loader = yaml.FullLoader)

VOCAB_SIZE = config['VOCAB_SIZE']
PATH = './datasets/Flicker8k_Dataset/'
BATCH_SIZE = config['BATCH_SIZE']
MAX_LEN = config['MAX_LEN']
NUM_ENCODER_LAYERS = config["NUM_ENCODER_LAYERS"]
NUM_DECODER_LAYERS = config["NUM_DECODER_LAYERS"]
DIM_FF = config["DIM_FF"]
NHEADS = config["N_HEADS"]
PATCH_SIZE = config["PATCH_SIZE"]



image_names = os.listdir(PATH)
image_text = dict()

data = open("datasets/Flickr8k.token.txt").read().split('\n')[:-1]
image_names = []
for i in tqdm(range(0, len(data), 5)):
    text = data[i].split('\t')[0][:-2]
    if not text.endswith('.jpg') :
        continue
    image_names.append(text)
    image_text[text] = []
    for j in range(5):
        image_text[text].append(data[i + j].split('\t')[-1])


with open('flickr.txt', 'w') as file :
    for i in image_text.values() :
        for j in range(5):
            file.write(i[j] + ' ')


yttm.BPE.train(data = 'flickr.txt', vocab_size = VOCAB_SIZE, model = 'bpe.model')
bpe = yttm.BPE(model = 'bpe.model')



dataset = ImageCaptionDataset(PATH, image_names, image_text, max_len = MAX_LEN)
train_dl = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)


model = Image2TextTransformer(
    nheads = NHEADS, 
    num_decoder_layers = NUM_DECODER_LAYERS,
    num_encoder_layers = NUM_ENCODER_LAYERS,
    dim_ff = DIM_FF,
    patch_size = PATCH_SIZE,
    vocab_size = VOCAB_SIZE )


trainer = L.Trainer(max_epochs = config['EPOCHS'])
trainer.fit(model, train_dl)


torch.save(model.state_dict(), "ImageCaption.pt")