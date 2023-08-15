namespace MinGptSharp

[<AutoOpen>]
module Utils =

    /// Pythonic range sequence.
    let range n = seq { 0 .. n - 1 }

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

    // https://github.com/dotnet/TorchSharp/discussions/982#discussioncomment-5759353
    type torch.Tensor with
        member t.GetSlice(startIdx: int64 option, endIdx: int64 option) =
            match startIdx, endIdx with
                | Some s, Some e -> t[torch.TensorIndex.Slice(s, e)]
                | Some s, None -> t[torch.TensorIndex.Slice(s)]
                | None, Some e -> t[torch.TensorIndex.Slice(stop=e)]
                | None, None -> t[torch.TensorIndex.Slice()]

        member t.SetSlice(startIdx: int64 option, endIdx: int64 option, v: torch.Tensor) =
            match startIdx, endIdx with
                | Some s, Some e -> t[torch.TensorIndex.Slice(s, e)] <- v
                | Some s, None -> t[torch.TensorIndex.Slice(s)] <- v
                | None, Some e -> t[torch.TensorIndex.Slice(stop=e)] <- v
                | None, None -> t[torch.TensorIndex.Slice()] <- v
