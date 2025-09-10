import numpy as np
import torch
import torch.nn as nn
import math
import open_clip
import json

from taming.models.attention import SpatialTransformer
from taming.models.utils import zero_module, conv_nd, normalization, checkpoint


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class PromptToken(nn.Module):
    """Class-conditional Prompt."""
    def __init__(self, vocab_size, embedding_size=768, hidden_size=768, length=16, hidden_dropout_prob=0.1, initializer_range=0.02):
        super(PromptToken, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.length = length
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        
        self.cls_embeds = nn.Embedding(vocab_size, embedding_size*length)
        self.dense = nn.Linear(embedding_size, hidden_size)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_embeds.weight, std=self.initializer_range)
        nn.init.trunc_normal_(self.dense.weight, std=self.initializer_range)

    def forward(self, x, deterministic=True):
        tokens = self.cls_embeds(x)
        tokens = tokens.view(-1, self.length, self.embedding_size)
        tokens = self.layer_norm(tokens)
        tokens = self.dense(tokens)
        
        if not deterministic:
            tokens = self.dropout(tokens)

        return tokens
    
class PromptLearner(nn.Module):
    def __init__(self, tokenizer, embeddings, length=16, ctx_dim=1024, targets=None, version="laion2b_s32b_b79k"):
        super().__init__()
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.length = length
        self.targets = targets
        self.target_token = PromptToken(vocab_size=len(self.targets), embedding_size=ctx_dim, hidden_size=ctx_dim, length=self.length)
        self.version = version

    def forward(self, texts):
        context = []

        target_indices = torch.tensor([self.targets.index(target) for target in texts]).unsqueeze(1).cuda()
        text_tokens = torch.cat([self.tokenizer(text) for text in texts]).cuda()

        target_embs = self.target_token(target_indices)
        if self.version == "laion2b_s32b_b79k":
            text_embs = self.embeddings(text_tokens)[:,:-self.length]
        else:
            text_embs = text_tokens[:,:-self.length]

        context = torch.cat([text_embs, target_embs], dim=1)

        return context.cuda()


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 seq_length=16, targets=None, freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        self.version = version
        
        if version == "laion2b_s32b_b79k":
            model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
            self.prompt_learner = PromptLearner(open_clip.tokenize, model.token_embedding, seq_length, ctx_dim=1024, targets=targets)
        else:
            model, preprocess = open_clip.create_model_from_pretrained(version)
            self.tokenizer = open_clip.get_tokenizer(version)
            self.prompt_learner = nn.Linear(512, 1024).cuda()

        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        
        if freeze: self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name: param.requires_grad = True
            else: param.requires_grad = False

    def forward(self, text):
        z = self.encode_with_transformer(text)
        return z

    def encode_with_transformer(self, text):
        if self.version == "laion2b_s32b_b79k":
            if text[0] == '':
                token = open_clip.tokenize(text).to(self.device)
                x = self.model.token_embedding(token).to(self.device)
            else:
                x = self.prompt_learner(text)
            
            x = x + self.model.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_final(x)
        else:
            token = self.tokenizer(text).to(self.device)
            x = self.model.encode_text(token).to(self.device)
            if text[0] != '':
                x = self.prompt_learner(x).to(self.device)
            
            x = x.unsqueeze(1)
        
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

class ZeroConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads, dim_head, zero_init=True, attn=False, context_dim=None, checkpoint=True):
        super().__init__()
        self.norm = normalization(in_ch)
        self.act = nn.ReLU()
        if zero_init:
            self.conv = zero_module(conv_nd(2, in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        else:
            self.conv = conv_nd(2, in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if attn:
            if zero_init:
                self.attn = zero_module(SpatialTransformer(out_ch, num_heads, dim_head, context_dim=context_dim, use_checkpoint=checkpoint))
            else:
                self.attn = SpatialTransformer(out_ch, num_heads, dim_head, context_dim=context_dim, use_checkpoint=checkpoint)
        else:
            self.attn = None    
        self.checkpoint = checkpoint

        if not zero_init:
            self.init_he()

    def init_he(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_normal_(p)


    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # h = self.skip(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x) #+ h
        if self.attn is not None:   
            x = self.attn(x, context)

        return x


class ZeroConvModule(nn.Module):
    def __init__(self, ddconfig):
        super().__init__()
        self.zero_convs = nn.ModuleList()
        ch = ddconfig['ch']
        ch_mult = ddconfig['ch_mult']
        num_resolutions = len(ch_mult)
        num_res_blocks = ddconfig['num_res_blocks']
        num_head_channels = ddconfig['num_head_channels']
        context_dim = ddconfig['context_dim']
        checkpoint = ddconfig['use_checkpoint']
        attn_res = ddconfig['attn_res']
        zero_init = ddconfig['zero_init']

        for i_level in range(num_resolutions):
            chs = ch * ch_mult[i_level]
            res = 2 ** i_level
            zero_conv_blocks = nn.ModuleList()
            for i_block in range(num_res_blocks+1):
                in_ch = chs
                out_ch = chs
                if i_level != 0 and i_level % 2 == 0 and i_block == 0:
                    in_ch = chs // 2
                elif i_level != 0 and i_level % 2 == 1 and i_block == num_res_blocks:
                    out_ch = chs * 2

                if num_head_channels == -1:
                    dim_head = ch // num_heads
                else:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                
                if res in attn_res: attn = True
                else: attn = False
                zero_conv_block = ZeroConvBlock(in_ch, out_ch, num_heads, dim_head, zero_init=zero_init,attn=attn, context_dim=context_dim, checkpoint=checkpoint)
                zero_conv_blocks.append(zero_conv_block)

            self.zero_convs.append(zero_conv_blocks)

    def forward(self, hs, context=None):
        zero_convs = [zero_conv for block in self.zero_convs for zero_conv in block]
        h_skips = [conv(h, context) for conv, h in zip(zero_convs, hs)]

        return h_skips