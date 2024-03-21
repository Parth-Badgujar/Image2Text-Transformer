import torch
import yaml
import youtokentome as yttm
import torchvision
import torch.nn.functional as F 
import torchvision.transforms as T
import os
from model import Image2TextTransformer
import argparse

parser = argparse.ArgumentParser(description = "Generate captions for images")
parser.add_argument("--images", type = str, help = "Path to the image")
args = parser.parse_args()
images_path = args.images

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


model.load_state_dict(torch.load("ImageCaption.pt"))


bpe = yttm.BPE(model = 'bpe.model')


@torch.inference_mode()
def generate(img):
    ids = torch.LongTensor([[2]])
    img_embeds = model.encode_image(img)
    while True :
        # print(bpe.decode([ids.tolist()[0][-1]])[0], end = '', flush  = True)
        new = model.decode_text(img_embeds, ids)[0,-1,:]
        new = torch.argmax(F.softmax(new, -1))
        ids = torch.cat([ids, new.unsqueeze(0).unsqueeze(0)], dim = 1)
        print(bpe.decode(ids.tolist())[0], end = '\r', flush=True)
        if new.item() == 3 :
            break
    ids = ids.cpu()
    # print(bpe.decode(ids.tolist())[0])

for img in os.listdir(images_path) :
    img_processed = T.functional.resize(torchvision.io.read_image(os.path.join(images_path, img)) / 255, (224, 224), antialias = True)
    print("Generating for image: ", img)
    generate(img_processed.unsqueeze(0))
    print()












