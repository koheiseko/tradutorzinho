import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Root Mean Squared Layer Normalization (RMSNorm).

        Args:
            theta:
            dim:
            eps:
            device:
            dtype:

        Atributos:
            weight:
            eps:
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = (x / rms) * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa a função de ativação Swish Linear Gated Unit (SwiGLU).

        Args:
            theta:
            dim:
            hidden_dim:
            device:
            dtype:

        Atributos:
            w1:
            w2:
            w3:
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, device=device, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
        """
        x1 = self.w1(x)
        values = x1 * torch.sigmoid(x1)

        gates = self.w3(x)

        return self.w2(values * gates)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        dim: int,
        context_length: int,
        device: torch.device | None = None,
    ):
        """
        Inicializa o Rotary Position Embedding (RoPE)

        RoPE incorpora informação posicional rotacionando pares consecutivos de componentes dos vetores de entrada. O ângulo usado em cada rotação depende tanto da posição do token quanto da frequência associada ao par de componentes

        Os valores de seno e cosseno são pré-calculados para todas as posições até "context_length" e armazenados em um buffer não treinável

        Args:
            theta: Base utilizada para definir as frequências das rotações
            dim: Dimensão dos vetores sobre os quais RoPE será aplicado. Deve ser um número par, pois os componentes são processados em pares
            max_seq_len: Maior quantidade de posições pré-calculadas no cache
            device: Device no qual o cache de frequências será criado. Quando "None", utiliza o dispositivo padrão do PyTorch

        Atributos:
            _freq_cis_cache: Buffer não persistente contendo os valores de cosseno
            e seno pré-calculados, com shape (2, context_length, dim // 2). O índice 0 contém os cossenos e o índice 1 contém os senos

        """
        super().__init__()

        self.register_buffer(
            "_freq_cis_cache",
            RotaryPositionalEmbedding._init_cache(
                context_length, dim, theta, device
            ),
            persistent=False,
        )

        self._freq_cis_cache: Float[
            torch.Tensor,
            "2 max_seq_length half_dim",
        ]

    @staticmethod
    def _init_cache(
        context_length: int,
        dim: int,
        theta: float,
        device: torch.device | None = None,
    ) -> Float[torch.Tensor, " 2 context_length half_dim"]:
        """
        Pré-calcula os senos e cossenos utilizados nas rotações

        Para cada posição e cada par de componentes, calcula uma frequência angular e armazena seu cosseno e seno. O cache é criado em "torch.float32" para preservar estabilidade numérica
        """
        if dim % 2 != 0:
            raise ValueError(
                f'O valor de "dim" deve ser par, mas recebeu {dim}.'
            )

        pairs_counts = (
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / dim
        )
        t = torch.arange(
            context_length,
            dtype=torch.float32,
            device=device,
        )
        freqs = theta**-pairs_counts

        freqs = einops.einsum(
            t,
            freqs,
            "t, f -> t f",
        )

        cos, sin = freqs.cos(), freqs.sin()

        return torch.stack((cos, sin))

    def forward(
        self,
        x: Float[
            torch.Tensor,
            "... sequence_length dim",
        ],
        token_positions: Int[
            torch.Tensor,
            "... sequence_length",
        ]
        | None = None,
    ) -> Float[
        torch.Tensor,
        "... sequence_length dim",
    ]:
        """
        Rotaciona os pares de componentes de acordo com suas posições.

        Quando "token_positions" não é fornecido, são usadas sequencialmente as posições de zero até "sequence_length - 1". Quando fornecido, o tensor permite selecionar posições específicas do cache

        Args:
            x: Tensor sobre o qual RoPE será aplicado
            token_positions: Índices das posições de cada token no intervalo

        Returns:
            Tensor rotacionado com o mesmo shape e dtype de "x".
        """
        if x.size(-1) % 2 != 0:
            raise ValueError(
                f"A última dimensão de `x` deve ser par, mas recebeu {x.size(-1)}."
            )

        x1, x2 = x[..., 0::2], x[..., 1::2]

        if token_positions is not None:
            cos, sin = self._freq_cis_cache[
                :,
                token_positions,
                :,
            ].unbind(0)

        else:
            seq_len = x.size(-2)

            if seq_len > self._freq_cis_cache.size(1):
                raise ValueError(
                    f"Comprimento da sequência ({seq_len}) excede "
                    f"max_seq_len ({self._freq_cis_cache.size(1)})."
                )

            cos, sin = self._freq_cis_cache[
                :,
                :seq_len,
                :,
            ].unbind(0)

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)

        return result


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        is_causal: bool,
        dropout: float,
        positional_encoder: RotaryPositionalEmbedding | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Multi Head Attention (MHA).

        Args:
            Args:
            d_model:
            num_heads:
            max_seq_len:
            theta:
            device:
            dtype:

        Atributos:
            head_dim:
            rope:
            w_q:
            w_k:
            w_v:
            w_o:
        """
        super().__init__()

        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.positional_encoder = positional_encoder

        self.q_proj = nn.Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(
            self.head_dim * num_heads, d_model, device=device, dtype=dtype
        )

    def forward(
        self,
        query: Float[torch.Tensor, "... queries d_model"],
        key: Float[torch.Tensor, "... keys d_model"],
        value: Float[torch.Tensor, "... keys d_model"],
        mask: Float[torch.Tensor, "... keys"] | None,
        token_positions: Int[torch.Tensor, "... keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_model"]:
        """


        Args:
            x:
            token_positions:
        """
        query = einops.rearrange(
            self.q_proj(query),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        )
        key = einops.rearrange(
            self.k_proj(key),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        )
        value = einops.rearrange(
            self.v_proj(value),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        )

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = einops.rearrange(
                    token_positions,
                    "... sequence_length -> ... 1 sequence_length",
                )

            query = self.positional_encoder(query, token_positions)
            key = self.positional_encoder(key, token_positions)

        B = query.size(0)
        L = query.size(-2)
        S = key.size(-2)

        attn_mask = None

        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("mask deve ser um tensor booleano.")

            if mask.shape != (B, S):
                raise ValueError(
                    f"Esperado mask com shape {(B, S)}, "
                    f"mas recebeu {tuple(mask.shape)}."
                )

            attn_mask = mask[:, None, None, :].to(query.device)

        if self.is_causal and attn_mask is not None:
            causal_mask = torch.ones(
                L, S, dtype=torch.bool, device=query.device
            ).tril()

            attn_mask = attn_mask & causal_mask

        attn = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal and attn_mask is None,
        )

        attn = einops.rearrange(
            attn,
            "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)",
        )

        output = self.o_proj(attn)

        return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None,
        dropout: float,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o bloco do Transformer.

        Args:
            d_model:
            d_ff:
            num_heads:
            max_seq_len:
            theta:
            eps:
            device:
            dtype:

        Atributos:
            ffn:
            mha:
            attn_norm:
            ffn_norm
        """
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            is_causal=False,
            dropout=dropout,
            positional_encoder=positional_encoder,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            dim=d_model, hidden_dim=d_ff, device=device, dtype=dtype
        )
        self.attn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )
        self.ffn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        mask: Float[torch.Tensor, "... seq_len"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """


        Args:
            x:
            token_positions:
        """
        x_attn_norm = self.attn_norm(x)
        x = x + self.mha(
            query=x_attn_norm,
            key=x_attn_norm,
            value=x_attn_norm,
            mask=mask,
            token_positions=token_positions,
        )

        x = x + self.ffn(self.ffn_norm(x))

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        positional_encoder: RotaryPositionalEmbedding | None,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o bloco do Transformer.

        Args:
            d_model:
            d_ff:
            num_heads:
            max_seq_len:
            theta:
            eps:
            device:
            dtype:

        Atributos:
            ffn:
            mha:
            attn_norm:
            ffn_norm
        """
        super().__init__()

        self.masked_mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            is_causal=True,
            dropout=dropout,
            positional_encoder=positional_encoder,
            device=device,
            dtype=dtype,
        )

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            is_causal=False,
            dropout=dropout,
            positional_encoder=None,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            dim=d_model, hidden_dim=d_ff, device=device, dtype=dtype
        )

        self.masked_attn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )
        self.attn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )
        self.ffn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        src_mask: Float[torch.Tensor, "... seq_len"],
        encoder_output: Float[torch.Tensor, "... seq_len d_model"],
        tgt_mask: Float[torch.Tensor, "... seq_len"],
        tgt_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """


        Args:
            x:
            token_positions:
        """

        x_masked_attn_norm = self.masked_attn_norm(x)
        x = x + self.masked_mha(
            query=x_masked_attn_norm,
            key=x_masked_attn_norm,
            value=x_masked_attn_norm,
            mask=tgt_mask,
            token_positions=tgt_positions,
        )

        x_attn_norm = self.attn_norm(x)
        x = x + self.mha(
            query=x_attn_norm,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        x = x + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        context_length: int,
        dropout: float,
        theta: float,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o modelo de Transformer.

        Args:
            d_model:
            d_ff:
            num_heads:
            max_seq_len:
            theta:
            eps:
            device:
            dtype:

        Atributos:
            tok_embeddings:
            transformer_blocks:
            norm:
            output:
        """
        super().__init__()

        self.src_embeddings = nn.Embedding(
            embedding_dim=d_model,
            num_embeddings=src_vocab_size,
            dtype=dtype,
            device=device,
        )
        self.tgt_embeddings = nn.Embedding(
            embedding_dim=d_model,
            num_embeddings=tgt_vocab_size,
            dtype=dtype,
            device=device,
        )

        d_head = d_model // num_heads

        self.positional_encoder = RotaryPositionalEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=theta,
            device=device,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    positional_encoder=self.positional_encoder,
                    dropout=dropout,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    positional_encoder=self.positional_encoder,
                    dropout=dropout,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.encoder_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )
        self.norm = RMSNorm(dim=d_model, eps=eps, device=device, dtype=dtype)
        self.output = nn.Linear(
            d_model, tgt_vocab_size, device=device, dtype=dtype
        )

    def forward(
        self,
        src: Int[torch.Tensor, "batch src_len"],
        tgt: Int[torch.Tensor, "batch tgt_len"],
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_positions: Int[torch.Tensor, "... seq_len"] | None = None,
        tgt_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        """


        Args:
            x:
            token_positions:
        """
        if src_positions is None:
            seq_len = src.size(-1)
            src_positions = torch.arange(seq_len, device=src.device)

        if tgt_positions is None:
            seq_len = tgt.size(-1)
            tgt_positions = torch.arange(seq_len, device=tgt.device)

        h_encoder = self.src_embeddings(src)
        h_decoder = self.tgt_embeddings(tgt)

        for encoder_block in self.encoder_blocks:
            h_encoder = encoder_block(
                x=h_encoder, mask=src_mask, token_positions=src_positions
            )

        h_encoder = self.encoder_norm(h_encoder)

        for decoder_block in self.decoder_blocks:
            h_decoder = decoder_block(
                x=h_decoder,
                src_mask=src_mask,
                encoder_output=h_encoder,
                tgt_mask=tgt_mask,
                tgt_positions=tgt_positions,
            )

        h_decoder = self.norm(h_decoder)
        output = self.output(h_decoder)

        return output
