namespace MinGptSharp

open System

open TorchSharp
open type TorchSharp.torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

[<AutoOpen>]
module TorchExt =
    let s (x : float) = x.ToScalar()

/// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
/// Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
type NewGELU() =
    inherit nn.Module<Tensor, Tensor>("NewGELU")
    override _.forward(x) =
        s 0.5 * x * (s 1.0 + torch.tanh(s (Math.Sqrt(2.0 / Math.PI)) * (x + s 0.044715 * torch.pow(x, s 3.0))))

type Config =
    {
        n_embd : int64
        n_head : int64
        attn_pdrop : float
        resid_pdrop : float
        block_size : int64
    }

#nowarn "25"   // allow pattern matching on arrays

/// A vanilla multi-head masked self-attention layer with a projection at the end.
/// It is possible to use torch.nn.MultiheadAttention here but I am including an
/// explicit implementation here to show that there is nothing too scary here.
type CausalSelfAttention(config) as self=
    inherit nn.Module<Tensor, Tensor>("CausalSelfAttention")

    do assert(config.n_embd % config.n_head = 0)
    // key, query, value projections for all heads, but in a batch
    let c_attn = nn.Linear(config.n_embd, 3L * config.n_embd)
    // output projection
    let c_proj = nn.Linear(config.n_embd, config.n_embd)
    // regularization
    let attn_dropout = nn.Dropout(config.attn_pdrop)
    let resid_dropout = nn.Dropout(config.resid_pdrop)
    // causal mask to ensure that attention is only applied to the left in the input sequence
    do self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
    let n_head = config.n_head
    let n_embd = config.n_embd

    do self.RegisterComponents()

    override _.forward(x) =
        let [| B; T; C |] = x.size() // batch size, sequence length, embedding dimensionality (n_embd)

        // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        let [| q; k; v |] = c_attn.forward(x).split(n_embd, dim=2)
        let k = k.view(B, T, n_head, C / n_head).transpose(1, 2) // (B, nh, T, hs)
        let q = q.view(B, T, n_head, C / n_head).transpose(1, 2) // (B, nh, T, hs)
        let v = v.view(B, T, n_head, C / n_head).transpose(1, 2) // (B, nh, T, hs)

        // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        let att = (q @@ k.transpose(-2, -1)) * s (1.0 / Math.Sqrt(float <| k.size(-1)))
        let att =
            let bias = self._internal_buffers["bias"]
            let mask = bias[Colon, Colon, Slice(stop=T), Slice(stop=T)]
            att.masked_fill((mask = tensor 0), s Double.NegativeInfinity)
        let att = softmax(att, dim = -1)
        let att = attn_dropout.forward(att)
        let y = att @@ v // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        let y = y.transpose(1, 2).contiguous().view(B, T, C) // re-assemble all head outputs side by side

        // output projection
        let y = resid_dropout.forward(c_proj.forward(y))
        y

/// an unassuming Transformer block
type Block(config) as self =
    inherit nn.Module<Tensor, Tensor>("Block")

    let ln_1 = nn.LayerNorm(config.n_embd)
    let attn = new CausalSelfAttention(config)
    let ln_2 = nn.LayerNorm(config.n_embd)
    let mlp =
        nn.ModuleDict<nn.Module<Tensor, Tensor>>(
            struct ("c_fc", nn.Linear(config.n_embd, 4L * config.n_embd)),
            struct ("c_proj", nn.Linear(4L * config.n_embd, config.n_embd)),
            struct ("act", new NewGELU()),
            struct ("dropout", nn.Dropout(config.resid_pdrop)))
    let m = mlp
    let mlpf = fun x -> m["dropout"].forward(m["c_proj"].forward(m["act"].forward(m["c_fc"].forward(x)))) // MLP forward

    do self.RegisterComponents()

    override _.forward(x) =
        let x = x + attn.forward(ln_1.forward(x))
        let x = x + mlpf(ln_2.forward(x))
        x
