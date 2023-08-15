namespace MinGptSharp

open TorchSharp
open type torch
open type utils.data
open FSharp.Core.Operators   // reclaim "int64" and other F# operators

/// Dataset for the Sort problem. E.g. for problem length 6:
/// Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
/// input:  0 0 2 1 0 1 0 0 0 1 1
/// output: I I I I I 0 0 0 1 1 2
/// where I is "ignore", as the transformer is reading the input sequence
type SortDataset(split, ?length, ?num_digits) as self =
    inherit Dataset()

    let length = defaultArg length 6
    let num_digits = defaultArg num_digits 3

    do assert(List.contains split ["train"; "test"])

    override _.Count with get() = 10000L

    member _.get_vocab_size() = num_digits

    member _.get_block_size() = self.Count * 2L - 1L

    override _.GetTensor(idx) =

        // use rejection sampling to generate an input example from the desired split
        let rec loop () =
            // generate some random integers
            let inp = torch.randint(int64 num_digits, size=[|self.Count|], dtype=torch.long)
            // half of the time let's try to boost the number of examples that 
            // have a large number of repeats, as this is what the model seems to struggle
            // with later in training, and they are kind of rare
            let reject =
                if torch.rand(1).item() < 0.5 then
                    let struct (unique, _, _) = inp.unique()
                    // too many unqiue digits, re-sample
                    unique.NumberOfElements > self.Count / 2L
                else false
            if reject then loop ()
            else
                // figure out if this generated example is train or test based on its hash
                let h = inp.GetHashCode()
                let inp_split = if h % 4 = 0 then "test" else "train" // designate 25% of examples as test
                if inp_split = split then
                    inp
                else loop ()

        let inp = loop ()
        
        // solve the task: i.e. sort
        let struct (sorted, _) = torch.sort(inp)
        let sol = sorted[0]

        // concatenate the problem specification and the solution
        let cat = torch.cat(ResizeArray [inp; sol], dim=0)

        // the inputs to the transformer will be the offset sequence
        let x = cat[.. cat.NumberOfElements - 1L].clone()
        let y = cat[1 ..].clone()
        // we only want to predict at output locations, mask out the loss at the input locations
        y[.. self.Count-1L] <- -1
        dict [ "x", x; "y", y] |> System.Collections.Generic.Dictionary

