namespace MinGptSharp

open System
open System.Collections.Generic

open TorchSharp
open type torch

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
        let cache = Dictionary<_, _>()
        fun x ->
            match cache.TryGetValue(x) with
                | true, v -> v
                | false, _ ->
                    let v = f x
                    cache.Add(x, v)
                    v

    /// Converts to scalar.
    let s (x : float) = x.ToScalar()

    /// Replacement for torch's @ operator.
    let (@@) a b = torch.matmul(a, b)

    /// Sets random seed.
    let set_seed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

type MinDataset = torch.utils.data.Dataset<Tensor * Tensor>

/// Minimal data loader.
type MinDataLoader(dataset : MinDataset, batch_size, ?shuffle, ?num_worker, ?drop_last) =
    inherit utils.data.DataLoader<Tensor * Tensor, Tensor * Tensor>(dataset, batch_size, MinDataLoader.Collate, ?shuffle=shuffle, ?num_worker=num_worker, ?drop_last=drop_last)

    static let collate f items (device : Device) =
        let tensors =
            items
                |> Seq.map (fun item ->
                    let (tensor : torch.Tensor) = f item
                    tensor.unsqueeze(0))
                |> Seq.toArray
        let tensor = torch.cat(tensors, 0)
        if tensor.device_type <> device.``type`` || tensor.device_index <> device.index then
            tensor.``to``(device)
        else tensor

    static member private Collate =
        Func<_, _, _>(fun pairs device ->
            let pairs = Seq.cache pairs
            collate fst pairs device,
            collate snd pairs device)

type ModelConfig =
    {
        model_type : string
        n_layer : int
        n_head : int
        n_embd : int
        vocab_size : int
        block_size : int
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
