import torch
import torch.nn as nn

from model.PseTNet.text_part.CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.PseTNet.text_part.CLIP.CLIP_tokenize import tokenize
from model.PseTNet.text_part.Transformer import Transformer, LayerNorm

_tokenizer = _Tokenizer()


def load_clip_text(state_dict_path):
    model = torch.jit.load(state_dict_path, map_location='cpu')
    state_dict = model.state_dict()

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64  # 8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))  # 12

    transformer = Transformer(
        width=transformer_width,
        layers=transformer_layers,
        heads=transformer_heads,
        attn_mask=None
    )
    token_embedding = nn.Embedding(vocab_size, transformer_width)
    positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
    ln_final = LayerNorm(transformer_width)
    text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    return transformer, transformer_width, token_embedding, positional_embedding, ln_final, text_projection


def text_params(state_dict_path):
    model = torch.jit.load(state_dict_path, map_location='cpu')
    state_dict = model.state_dict()

    text_state_dict = {}
    for k in state_dict.keys():
        if 'visual' in k:
            continue
        text_state_dict['text_encoder.' + k] = state_dict[k]
    return text_state_dict


class Custom_Text_Prompt(nn.Module):
    def __init__(self, n_vector: int, vector_dim: int, keywords: str, token_embedding):
        super().__init__()
        text_vector = torch.empty(n_vector, vector_dim)
        nn.init.normal_(text_vector, std=0.02)
        self.text_vector = nn.Parameter(text_vector)
        vector_learn = " ".join(["-"] * n_vector)
        print(f"Number of learnable text vector: {n_vector}")

        text_prompt = vector_learn + " " + keywords + "."
        print(f'Initial text prompt: "{text_prompt}"')

        tokenized_prompts = tokenize(text_prompt)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        self.register_buffer("start_token", embedding[:, :1, :])
        self.register_buffer("end_token", embedding[:, 1 + n_vector:, :])

    def forward(self):
        text_vector = self.text_vector
        if text_vector.dim() == 2:
            text_vector = text_vector.unsqueeze(0).expand(1, -1, -1)
        prompts = torch.cat([self.start_token, text_vector, self.end_token], dim=1)

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, transformer, positional_embedding, ln_final):
        super().__init__()
        self.transformer = transformer
        self.positional_embedding = positional_embedding
        self.ln_final = ln_final

    def forward(self, prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x_features = x

        return x_features


class CustomCLIP(nn.Module):
    def __init__(self, n_vector, keywords, clip_params_path):
        super().__init__()
        transformer, transformer_width, token_embedding, positional_embedding, \
            ln_final, text_projection = load_clip_text(clip_params_path)
        self.prompt_learner = Custom_Text_Prompt(n_vector, transformer_width, keywords, token_embedding)
        self.text_encoder = TextEncoder(transformer, positional_embedding, ln_final)

    def forward(self):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts)

        return text_features
