namespace MinGptSharp

open TorchSharp
open type torch
open type utils.data

/// Dataset for the Sort problem. E.g. for problem length 6:
/// Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
/// input:  0 0 2 1 0 1 0 0 0 1 1
/// output: I I I I I 0 0 0 1 1 2
/// where I is "ignore", as the transformer is reading the input sequence
type SortDataset(split, ?length, ?num_digits) =
    inherit Dataset()

    let length = defaultArg length 6
    let num_digits = defaultArg num_digits 3

    do assert(List.contains split ["train"; "test"])

    override _.Count with get() = 10000L

    member _.get_vocab_size() = num_digits

    override _.GetTensor(index) =
        System.Collections.Generic.Dictionary<_, _>()
