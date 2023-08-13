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
