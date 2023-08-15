﻿namespace MinGptSharp

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
type SortDataset(split, ?length, ?num_digits) =
    inherit Dataset()

    let length = defaultArg length 6
    let num_digits = defaultArg num_digits 3

    do assert(List.contains split ["train"; "test"])

    let nTensorDicts = 10000

    let makeTensorDict () =

        // use rejection sampling to generate an input example from the desired split
        let rec loop () =
            // generate some random integers
            let inp = torch.randint(int64 num_digits, size=[|int64 length|], dtype=torch.long)
            // half of the time let's try to boost the number of examples that 
            // have a large number of repeats, as this is what the model seems to struggle
            // with later in training, and they are kind of rare
            let reject =
                if torch.rand(1).item() < 0.5f then
                    let struct (unique, _, _) = inp.unique()
                    // too many unqiue digits, re-sample
                    unique.NumberOfElements > int64 length / 2L
                else false
            if reject then loop ()
            else
                // figure out if this generated example is train or test based on its hash
                let inp_split = if torch.rand(1).item() < 0.25f then "test" else "train" // designate 25% of examples as test
                if inp_split = split then
                    inp
                else loop ()

        let inp = loop ()
        
        // solve the task: i.e. sort
        let struct (sol, _) = torch.sort(inp)

        // concatenate the problem specification and the solution
        let cat = torch.cat(ResizeArray [inp; sol], dim=0)

        // the inputs to the transformer will be the offset sequence
        let x = cat[.. cat.NumberOfElements - 1L].clone()
        let y = cat[1 ..].clone()
        // we only want to predict at output locations, mask out the loss at the input locations
        y[.. int64 length - 1L] <- -1
        dict [ "x", x; "y", y ] |> System.Collections.Generic.Dictionary

    let tensorDicts =
        Array.init nTensorDicts (fun _ -> makeTensorDict ())

    override _.Count with get() = nTensorDicts

    member _.get_vocab_size() = num_digits

    member _.get_block_size() = length * 2 - 1

    override _.GetTensor(idx) =
        tensorDicts[int idx]

module Program =

    set_seed 0L

    let train_dataset = new SortDataset("train")
    let test_dataset = new SortDataset("test")
    let x, y =
        let dict = train_dataset.GetTensor(0L)
        dict["x"], dict["y"]
    for a, b in Seq.zip (x.data<int64>()) (y.data<int64>()) do
        printfn $"{a} {b}"

    let model_config =
        {
            GPT.get_default_config() with
                model_type = "gpt-nano"
                vocab_size = train_dataset.get_vocab_size()
                block_size = train_dataset.get_block_size()
        }
    let model = new GPT(model_config)

    let train_config =
        {
            Trainer.get_default_config() with
                learning_rate = 5e-4 // the model we're using is so small that we can go a bit faster
                max_iters = 2000
                num_workers = 0
        }
    let trainer = Trainer(train_config, model, train_dataset)
    trainer.run ()
