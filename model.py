import torch
import torch.nn as nn
import lightning as L 
import torch.nn.functional as F

class Image2TextTransformer(L.LightningModule):
    def __init__(self, nheads, num_encoder_layers, num_decoder_layers, dim_ff, patch_size, vocab_size):
        super().__init__()
        self.h = self.w = 224
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.dmodel = (patch_size * patch_size * 3)
        self.embeddimg = nn.Embedding(vocab_size, self.dmodel)
        self.model = nn.Transformer(
            num_decoder_layers = num_decoder_layers,
            num_encoder_layers = num_encoder_layers,
            nhead = nheads,
            dim_feedforward = dim_ff,
            d_model = self.dmodel,
            batch_first = True
        )
        self.proj = nn.Linear(self.dmodel, vocab_size)
    def img_to_patch(self, x, patch_size):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1,2)
        x = x.flatten(2,4)
        return x
    def get_padding_mask(self, sen):
        return (sen == 0)
    def forward(self, img, tgt):
        src = self.img_to_patch(img, self.patch_size)
        tgt_padding_mask = self.get_padding_mask(tgt)
        tgt = self.embeddimg(tgt)
        B, T, C = tgt.shape
        tgt_mask = self.model.generate_square_subsequent_mask(T, device = self.device)
        out = self.model(src, tgt, tgt_key_padding_mask = tgt_padding_mask, tgt_mask = tgt_mask)
        out = self.proj(out)
        return out
    def encode_image(self, img):
        img = self.img_to_patch(img,  self.patch_size)
        embeds = self.model.encoder(img)
        return embeds
    def decode_text(self, img_embeds, text_tokens):
        tgt_padding_mask = self.get_padding_mask(text_tokens)
        tgt = self.embeddimg(text_tokens)
        B, T, C = tgt.shape
        tgt_mask = self.model.generate_square_subsequent_mask(T, device = self.device)
        out = self.model.decoder(tgt, img_embeds, tgt_key_padding_mask = tgt_padding_mask, tgt_mask = tgt_mask)
        out = self.proj(out)
        return out
    def training_step(self, batch, batch_idx):
        img, src, tgt = batch
        out = self(img, src)
        out = out.view(-1, self.vocab_size)
        loss = F.cross_entropy(out, tgt.view(-1), ignore_index = 0)
        self.log("Loss", loss.item(), on_step = True, on_epoch = True, prog_bar = True)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = 3e-4)