namespace MinGptSharp

open TorchSharp
open type torch
open type utils.data
open type TensorIndex
open FSharp.Core.Operators   // reclaim "int64" and other F# operators

/// Dataset for the Sort problem. E.g. for problem length 6:
/// Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
/// input:  0 0 2 1 0 1 0 0 0 1 1
/// output: I I I I I 0 0 0 1 1 2
/// where I is "ignore", as the transformer is reading the input sequence
type SortDataset(split, ?length, ?num_digits) =
    inherit MinDataset()

    let length = defaultArg length 6
    let num_digits = defaultArg num_digits 3

    do assert(List.contains split ["train"; "test"])

    let nTensorPairs = 10000

    let makeTensorPair _ =

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
        let sol = torch.sort(inp) |> fstv

        // concatenate the problem specification and the solution
        let cat = torch.cat([|inp; sol|], dim=0)

        // the inputs to the transformer will be the offset sequence
        let x = cat[Slice(stop = -1)].clone()
        let y = cat[Slice(1)].clone()
        // we only want to predict at output locations, mask out the loss at the input locations
        y[Slice(stop=length-1)] <- tensor -1
        x, y

    let tensorPairs =
        Array.init nTensorPairs makeTensorPair

    member _.Length = length

    override _.Count with get() = nTensorPairs

    member _.get_vocab_size() = num_digits

    member _.get_block_size() = length * 2 - 1

    override _.GetTensor(idx) =
        tensorPairs[int idx]

module Program =

    set_seed 0L

    let train_dataset = new SortDataset("train")
    let test_dataset = new SortDataset("test")
    let x, y = train_dataset.GetTensor(0L)
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

    let batch_end_callback progress =
        if progress.iter_num % 100 = 0 then
            printfn $"iter_dt {progress.iter_dt.TotalMilliseconds:f2}ms; iter {progress.iter_num}: train loss {progress.loss}"
    trainer.set_callback "on_batch_end" batch_end_callback

    trainer.run ()

    let getSeq (t : Tensor) =
        t.data<int64>()
            |> Seq.map (int >> string)
            |> String.concat ""

    let eval_split split max_batches =
        let dataset = (Map ["train", train_dataset; "test", test_dataset])[split]
        let n = train_dataset.Length
        let loader = new MinDataLoader(dataset, batch_size=100, num_worker=0, drop_last=false)
        let results, _ =
            let pairs =
                max_batches
                    |> Option.map (fun max -> Seq.truncate max loader)
                    |> Option.defaultValue loader
            (([], 0), pairs)
                ||> Seq.fold (fun (results, mistakes_printed_already) (x, y) ->
                    let x = x.``to``(trainer.Device)
                    let y = y.``to``(trainer.Device)
                    // isolate the input pattern alone
                    let inp = x[Colon, Slice(stop=n)]
                    let sol = y[Colon, Slice(-n)]
                    // let the model sample the rest of the sequence
                    let cat = model.generate(inp, n, do_sample=false) // using greedy argmax, not sampling
                    let sol_candidate = cat[Colon, Slice(n)] // isolate the filled in sequence
                    // compare the predicted sequence to the true sequence
                    let correct = (torch.eq(sol, sol_candidate)).all(1).cpu() // Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
                    ((results, mistakes_printed_already), rangel(x.size(0)))
                        ||> Seq.fold (fun (results, mistakes_printed_already) i ->
                            let results = int(correct[i]) :: results
                            let mistakes_printed_already =
                                if (not (correct[i].item<bool>())) && mistakes_printed_already < 5 then // only print up to 5 mistakes to get a sense
                                    let get (t : Tensor) = getSeq t[i]
                                    printfn "GPT claims that %s sorted is %s but gt is %s" (get inp) (get sol_candidate) (get sol)
                                    mistakes_printed_already + 1
                                else mistakes_printed_already
                            results, mistakes_printed_already))

        let results = Seq.toArray results
        let rt = torch.tensor(results, dtype=torch.float)
        printfn "%s final score: %.0f/%d = %.2f%% correct"
            split (rt.sum().item<float32>()) results.Length (100.0f * rt.mean().item<float32>())
        rt.sum()

    // run a lot of examples from both train and test through the model and verify the output correctness
    using (torch.no_grad()) (fun _ ->
        let train_score = eval_split "train" (Some 50)
        let test_score  = eval_split "test"  (Some 50)
        ())

    // let's run a random given sequence through the model as well
    let n = train_dataset.Length
    let inp = torch.tensor(array2D [[0; 0; 2; 1; 0; 1]], dtype=torch.long).``to``(trainer.Device)
    assert(inp[0].NumberOfElements = n)
    using (torch.no_grad()) (fun _ ->
        let cat = model.generate(inp, n, do_sample=false)
        let sol = torch.sort(inp[0]) |> fstv
        let sol_candidate = cat[Colon, Slice(n)]
        printfn "input sequence  : %s" (getSeq inp)
        printfn "predicted sorted: %s" (getSeq sol_candidate)
        printfn "gt sort         : %s" (getSeq sol)
        printfn "matches         : %A" <| torch.eq(sol, sol_candidate).all().item<bool>())
