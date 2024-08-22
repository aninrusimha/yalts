import torch
from torch import nn
from flash_attn import flash_attn_func


class AttentionModule(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_head):
        super(AttentionModule, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_q_heads = n_q_heads
        # change: handle n_kv_heads in arguments.py
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_head
        self.qkv_linear = nn.Linear(
            d_model, (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False
        )

        self.out_linear = nn.Linear(self.n_q_heads * self.d_head, d_model, bias=False)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()

        q, k, v = self.split(self.qkv_linear(x))

        # attention_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, causal=True).view(batch_size, seq_len, self.n_q_heads * self.head_dim)
        attention_output = flash_attn_func(q, k, v, causal=True).view(
            batch_size, seq_len, self.n_q_heads * self.head_dim
        )
        #  attention_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, causal=True).view(batch_size, seq_len, self.n_q_heads * self.head_dim)

        attention_output = self.out_linear(attention_output)

        return attention_output

    def split(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(
            batch_size, seq_len, self.n_q_heads + 2 * self.n_kv_heads, self.head_dim
        )
        q, k, v = torch.split(
            x, [self.n_q_heads, self.n_kv_heads, self.n_kv_heads], dim=2
        )
        return q, k, v


class MLPModule(nn.Module):
    def __init__(self, d_model, d_ff, activation=nn.GELU):
        super(MLPModule, self).__init__()
        self.ff_in = nn.Linear(d_model, d_ff, bias=False)
        self.ff_out = nn.Linear(d_ff, d_model, bias=False)
        self.act = activation()

    def forward(self, x):
        x = self.ff_in(x)
        x = self.act(x)
        x = self.ff_out(x)
        return x


class GLUModule(nn.Module):
    def __init__(self, d_model, d_ff, activation=nn.GELU):
        super(GLUModule, self).__init__()
        self.ff_in = nn.Linear(d_model, d_ff, bias=False)
        self.ff_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ff_out = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation()

    def forward(self, x):
        mid = self.ff_in(x)
        gate = self.activation(self.ff_gate(x))
        return self.ff_out(mid * gate)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, n_kv_heads, d_head, d_ff, glu):
        super(TransformerBlock, self).__init__()
        self.attn = AttentionModule(d_model, n_heads_q, n_kv_heads, d_head)
        if not glu:
            self.ffn = MLPModule(d_model, d_ff)
        else:
            self.ffn = GLUModule(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model, bias=False)

    def forward(self, x):
        resid = x
        attn_out = self.attn(self.ln1(x))
        resid = x + attn_out
        ffn_out = self.ffn(self.ln2(resid))
        resid = x + ffn_out
        return resid


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_ff = args.d_ff

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        # TODO: add RoPE or abs pos embeddings

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    args.d_model,
                    args.n_q_heads,
                    args.n_kv_heads,
                    args.d_head,
                    args.d_ff,
                    args.glu,
                )
                for _ in range(args.n_layers)
            ]
        )
        self.fc = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)

        for i in range(self.n_layers):
            x = self.transformer_blocks[i](x)

        x = self.fc(x)
        return x
