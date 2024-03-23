import argparse
parser = argparse.ArgumentParser(description = "Generate captions for images")
parser.add_argument("--images", type = str, help = "Path to the images")
parser.add_argument("--temp", type = float, help = "Temperature for sampling")
parser.add_argument("--model", type = str, help = 'Name of the pytorch model file', default = "ImageCaption.pt")
parser.add_argument("--bpe", type = str, help = "Name of the BPE model", default = "bpe.model")
args = parser.parse_args()
images_path = args.images


import torch
import yaml
import youtokentome as yttm
import torchvision
import torch.nn.functional as F 
import torchvision.transforms as T
import os
from model import Image2TextTransformer


config = yaml.load(open("config.yaml", "r"), Loader = yaml.FullLoader)

NUM_ENCODER_LAYERS = config["NUM_ENCODER_LAYERS"]
NUM_DECODER_LAYERS = config["NUM_DECODER_LAYERS"]
DIM_FF = config["DIM_FF"]
NHEADS = config["N_HEADS"]
VOCAB_SIZE = config["VOCAB_SIZE"]
PATCH_SIZE = config["PATCH_SIZE"]
PATH = "datasets/Flicker8k_Dataset/"


model = Image2TextTransformer(
    nheads = NHEADS, 
    num_decoder_layers = NUM_DECODER_LAYERS,
    num_encoder_layers = NUM_ENCODER_LAYERS,
    dim_ff = DIM_FF,
    patch_size = PATCH_SIZE,
    vocab_size = VOCAB_SIZE
)

try :
    model.load_state_dict(torch.load(args.model))
    model.eval()
    bpe = yttm.BPE(model = args.bpe)
except Exception as e  :
    print("Error loading models :", e)

@torch.inference_mode()
def generate(img):
    ids = torch.LongTensor([[2]])
    img_embeds = model.encode_image(img)
    while True :
        new = model.decode_text(img_embeds, ids)[0,-1,:]
        new = torch.multinomial(F.softmax(new / args.temp, -1), 1) 
        ids = torch.cat([ids, new.unsqueeze(0)], dim = 1)
        print(bpe.decode(ids.tolist())[0], end = '\r', flush=True)
        if new.item() == 3 :
            break

for img in os.listdir(images_path) :
    img_processed = T.functional.resize(torchvision.io.read_image(os.path.join(images_path, img)) / 255, (config['HEIGHT'], config["WIDTH"]), antialias = True)
    print("Generating for image: ", img)
    generate(img_processed.unsqueeze(0))
    print()












