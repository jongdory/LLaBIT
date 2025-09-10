import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from taming.main import instantiate_from_config

from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import Encoder, Decoder, nonlinearity
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.losses.lpips import LPIPS

from taming.models.modules import ZeroConvModule

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)
    intersection = (probs_flat * targets_flat).sum(-1)
    union = probs_flat.sum(-1) + targets_flat.sum(-1)
    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQModel2(VQModel):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False):
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         n_embed=n_embed,
                         embed_dim=embed_dim,
                         ckpt_path=ckpt_path,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         remap=remap,
                         sane_index_shape=sane_index_shape)
        
        self.encoder = Encoder2(**ddconfig)
        self.decoder = Decoder2(**ddconfig)

        self.zero_convs_tr = ZeroConvModule(ddconfig)
        self.zero_convs = ZeroConvModule(ddconfig)
        self.seg_layer = SegLayer(ddconfig['out_ch'], num_classes=1)
        
        self.percept_loss = LPIPS().eval()
        self.tr_loss = instantiate_from_config(lossconfig)
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.instantiate_cond_stage(ddconfig.cond_stage_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        self.cond_stage_model.train = disabled_train
        for name, param in self.cond_stage_model.named_parameters():
            if "prompt_learner" in name: param.requires_grad = True
            else: param.requires_grad = False

    def compute_tr_loss(self, batch, cond, optimizer_idx=0):
        xsrc = self.get_input(batch, "source_img")
        x_trg = self.get_input(batch, "target_img")
        toks = batch["target_tokens"]
        xrec = self(xsrc, toks, cond)
        
        rec_loss = torch.abs(x_trg.contiguous() - xrec.contiguous())
        percept_loss = self.percept_loss(x_trg.contiguous(), xrec.contiguous())
        nll_loss = rec_loss + percept_loss
        loss = torch.mean(nll_loss)

        self.log("train/rec_loss", rec_loss.clone().detach().mean(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/perceptloss", percept_loss.clone().detach().mean(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def compute_seg_loss(self, batch, cond):
        xsrc = self.get_input(batch, "source_img")
        xseg = self.get_input(batch, "target_img")
        toks = batch["target_tokens"]
        xrec = self(xsrc, toks, cond, mode="seg")
        xpred = self.seg_layer(xrec)

        dice = dice_loss(xpred, xseg)
        bce = self.bceloss(xpred, xseg)
        loss = dice + bce
        self.log("train/diceloss", dice,
                prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/bceloss", bce,
                prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/segloss", loss,
                prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        target = batch["target"]
        cond = self.cond_stage_model(target)

        tasks = batch["task"] 
        if "tr" in tasks:
            total_loss = self.compute_tr_loss(batch, cond)
        if "seg" in tasks:
            total_loss = self.compute_seg_loss(batch, cond)

        return total_loss


    def validation_step(self, batch, batch_idx):
        xsrc = self.get_input(batch, "source_img")
        target = batch[f"target"]
        cond = self.cond_stage_model(target)

        return self.log_dict
    
    def encode(self, x):
        _, hs = self.encoder(x)
        return hs

    def decode(self, quant, hs):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, hs)
        return dec

    def forward(self, input, info, cond, mode="tr"):
        hs = self.encode(input)
        bs = input.shape[0]
        quant = self.quantize.get_codebook_entry(info, shape=(bs,12,12,-1)).cuda()
        if mode == "tr":
            hskips = self.zero_convs_tr(hs, cond)
        else:
            hskips = self.zero_convs(hs, cond)
        dec = self.decode(quant, hskips)
        return dec
    
    def log_images(self, batch, **kwargs):
        log = dict()

        xsrc = self.get_input(batch, "source_img").to(self.device)
        target = batch[f"target"]
        cond = self.cond_stage_model(target).to(self.device)
        log["source"] = xsrc
        toks = batch[f"target_tokens"].to(self.device)
        xtrg = self.get_input(batch, "target_img").to(self.device)

        if "tr" in batch["task"]:    
            xrec = self(xsrc, toks, cond)
            log["target"] = xtrg
            log["reconstruction"] = xrec
        else:
            xrec = self(xsrc, toks, cond, mode="seg")
            log["target"] = xtrg
            log["reconstruction"] = xrec
            xpred = self.seg_layer(xrec)
            log["seg_preds"] = xpred

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.zero_convs_tr.parameters()) +
                               list(self.zero_convs.parameters()) +
                               list(self.seg_layer.parameters()) +
                               list(self.cond_stage_model.prompt_learner.parameters()), 
                               lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        
        return [opt], []


class SegLayer(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super(SegLayer, self).__init__()
        hd = 64
        self.conv = nn.Conv2d(in_channels, hd, kernel_size=1)
        self.norm_out = nn.InstanceNorm2d(hd)
        self.conv_out = nn.Conv2d(hd, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        
        return x


class Encoder2(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, hs


class Decoder2(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z, h_skips):
        self.last_z_shape = z.shape
        temb = None
        hs_idx = len(h_skips) - 1
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling and skip connection
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = h + h_skips[hs_idx]
                hs_idx -= 1
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    