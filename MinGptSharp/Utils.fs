﻿namespace MinGptSharp

[<AutoOpen>]
module Utils =

    /// Pythonic range sequence.
    let range n = seq { 0 .. n - 1 }

    /// Pythonic range sequence.
    let rangel n = seq { 0L .. n - 1L }

    /// Integer power.
    let powi x y =
        Seq.replicate y x |> Seq.fold (*) 1

    /// First item of a value tuple.
    let fstv (struct (x, _)) = x

    /// Second item of a value tuple.
    let sndv (struct (_, y)) = y

    /// Memoizes the given function.
    /// http://www.fssnip.net/mW/title/memoize-
    let memoize f =
        let cache = System.Collections.Generic.Dictionary<_, _>()
        fun x ->
            match cache.TryGetValue(x) with
                | true, v -> v
                | false, _ ->
                    let v = f x
                    cache.Add(x, v)
                    v

    open TorchSharp

    // replacement for @ operator
    let (@@) a b = torch.matmul(a, b)

    let set_seed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

type ModelConfig =
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

type TrainerConfig =
    {
        device : string
        num_workers : int
        max_iters : int
        batch_size : int
        learning_rate : float
        betas : float * float
        weight_decay : float
        grad_norm_clip : float
    }
