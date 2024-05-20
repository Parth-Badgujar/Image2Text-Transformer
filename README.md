## Simmple BLIP

* First we split the image into patches similar to `Vision-Transformer` and send it through the encoder which generate embeddings of the image
* Then we decode the text autoregressively with the decoder where the encoded patches are `cross-attended` with text embeddings 

## Usage 

* Download the `flickr8k` dataset using 
```
./flick8k.sh
```
* Then modify the `config.yaml`, accordingly 
```yaml
EPOCHS: 1
BATCH_SIZE: 64

N_HEADS : 4
NUM_ENCODER_LAYERS : 4
NUM_DECODER_LAYERS : 4
DIM_FF : 256
PATCH_SIZE: 16
HEIGHT : 224
WIDTH : 224

MAX_LEN : 128
VOCAB_SIZE : 128
```
* Run the `train.py` script, I am using `pytorch-lightning` so it will automatically detect any GPUs if present 
```
python3 train.py
``` 

* Use the `eval.py` to run the model on custom images 
```
usage: eval.py [-h] [--images IMAGES] [--temp TEMP] [--model MODEL] [--bpe BPE]

Generate captions for images

options:
  -h, --help       show this help message and exit
  --images IMAGES  Path to the images
  --temp TEMP      Temperature for sampling
  --model MODEL    Name of the pytorch model file
  --bpe BPE        Name of the BPE model
```
