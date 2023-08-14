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

type CfgNode =
    {
        model_type : string
        n_layer : int
        n_head : int64
        n_embd : int64
        vocab_size : int64
        block_size : int64
        embd_pdrop : float
        resid_pdrop : float
        attn_pdrop : float
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

/// GPT Language Model
type GPT(config) as self =
    inherit nn.Module<Tensor, Tensor>("GPT")

    static let get_default_config () =
        {
            // either model_type or (n_layer, n_head, n_embd) must be given in the config
            model_type = "gpt"
            n_layer = -1
            n_head = -1L
            n_embd =  -1L
            // these options must be filled in externally
            vocab_size = -1L
            block_size = -1L
            // dropout hyperparameters
            embd_pdrop = 0.1
            resid_pdrop = 0.1
            attn_pdrop = 0.1
        }

    do
        assert(config.vocab_size > 0)
        assert(config.block_size > 0)
    let block_size = config.block_size

    let type_given = String.IsNullOrWhiteSpace(config.model_type) |> not
    let params_given = config.n_layer > 0 && config.n_head > 0 && config.n_embd > 0
    do assert (type_given <> params_given) // exactly one of these (XOR)
    let config =
        if type_given then
            // translate from model_type to detailed configuration
            match config.model_type with
                // names follow the huggingface naming conventions
                // GPT-1
                | "openai-gpt" ->  { config with n_layer=12; n_head=12; n_embd= 768 } // 117M params
                // GPT-2 configs
                | "gpt2" ->        { config with n_layer=12; n_head=12; n_embd= 768 } // 124M params
                | "gpt2-medium" -> { config with n_layer=24; n_head=16; n_embd=1024 } // 350M params
                | "gpt2-large" ->  { config with n_layer=36; n_head=20; n_embd=1280 } // 774M params
                | "gpt2-xl"  ->    { config with n_layer=48; n_head=25; n_embd=1600 } // 1558M params
                // Gophers
                | "gopher-44m" ->  { config with n_layer= 8; n_head=16; n_embd= 512 }
                // (there are a number more...)
                // I made these tiny models up
                | "gpt-mini" ->    { config with n_layer= 6; n_head= 6; n_embd= 192 }
                | "gpt-micro" ->   { config with n_layer= 4; n_head= 4; n_embd= 128 }
                | "gpt-nano" ->    { config with n_layer= 3; n_head= 3; n_embd=  48 }
        else config

    let transformer =
        nn.ModuleDict<nn.Module>(
            struct ("wte", nn.Embedding(config.vocab_size, config.n_embd)),
            struct ("wpe", nn.Embedding(config.block_size, config.n_embd)),
            struct ("drop", nn.Dropout(config.embd_pdrop)),
            struct ("h", nn.ModuleList([| for _ in range(config.n_layer) -> new Block(config) |])),
            struct ("ln_f", nn.LayerNorm(config.n_embd)))
    let lm_head = nn.Linear(config.n_embd, config.vocab_size, hasBias=false)

    let _init_weights(module' : nn.Module) =
        match module' with
            | :? Modules.Linear as linear ->
                torch.nn.init.normal_(linear.weight, mean=0.0, std=0.02) |> ignore
                if isNull linear.bias |> not then
                    torch.nn.init.zeros_(linear.bias) |> ignore
            | :? Modules.Embedding as embedding ->
                torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02) |> ignore
            | :? Modules.LayerNorm as norm ->
                torch.nn.init.zeros_(norm.bias) |> ignore
                torch.nn.init.ones_(norm.weight) |> ignore

    do
        // init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(_init_weights) |> ignore
        for pn, p in self.named_parameters() do
            if pn.EndsWith("c_proj.weight") then
                torch.nn.init.normal_(p, mean=0.0, std=0.02/Math.Sqrt(2.0 * float config.n_layer)) |> ignore

        // report number of parameters (note we don't count the decoder parameters in lm_head)
        let n_params = Seq.sum [ for p in transformer.parameters() -> p.numel() ]
        printfn "number of parameters: %.2fM" (float n_params/1.0e6)
