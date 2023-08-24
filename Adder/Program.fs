namespace MinGptSharp

(*
Trains a GPT to add n-digit numbers.
*)

open System

open TorchSharp
open type torch
open type utils.data
open type TensorIndex
open FSharp.Core.Operators   // reclaim "int64" and other F# operators

type AdditionDatasetConfig =
    {
        ndigit : int
    }

/// Creates n-digit addition problems. For example, if n=2, then an example
/// addition problem would be to add 85 + 50 = 135. This problem would be
/// represented as the following string for the GPT:
/// 
/// "8550531"
/// 
/// This is because:
/// - we are discarding the + and =, which are not necessary. We just encode the digits
///     of the input numbers concatenated together.
/// - the result 135 is encoded backwards to make the addition easier to learn for the
///     GPT model, because of how the addition algorithm works.
/// 
/// As one more example, the problem 6 + 39 = 45 would be encoded as:
/// 
/// "0639054"
/// 
/// where you will notice that we are padding with zeros to make sure that we always
/// produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
/// At test time, we will feed in an addition problem by giving the first 2n digits,
/// and hoping that the GPT model completes the sequence with the next (n+1) digits
/// correctly.
type AdditionDataset(config, split (*train/test*)) =
    inherit MinDataset()

    // split up all addition problems into either training data or test data
    let ndigit = config.ndigit
    do assert(ndigit <= 3) // "the lines below would be very memory inefficient, in future maybe refactor to support"
    let num = powi (powi 10 ndigit) 2 // total number of possible addition problems with ndigit numbers
    let rng = new torch.Generator()
    do rng.manual_seed(1337) |> ignore
    let perm = torch.randperm(num, generator=rng)
    let num_test = min ((num / 5)) 500 // 20% of the whole dataset, or only up to 500
    let ixes =
        if split = "test" then perm[Slice(stop=num_test)]
        else perm[Slice(num_test)]        

    static member get_default_config() =
        {
            ndigit = 2
        }

    member _.get_vocab_size() =
        10 // digits 0..9

    member _.get_block_size() =
        // a,b,a+b, and +1 due to potential carry overflow,
        // but then also -1 because very last digit doesn't ever plug back
        // as there is no explicit <EOS> token to predict, it is implied
        3 * ndigit + 1 - 1

    override _.Count with get() = ixes.NumberOfElements

    override _.GetTensor(idx) =
        // given a problem index idx, first recover the associated a + b
        let idx = ixes[idx].item<int64>() |> int
        let nd = powi 10 ndigit
        let a = idx / nd
        let b = idx %  nd
        // calculate the "label" of the addition problem a + b
        let c = a + b
        // encode the digits of a, b, c into strings
        let fmt (n : int) (x : int) = x.ToString(String.Format("d{0}", n))
        let astr = fmt ndigit a
        let bstr = fmt ndigit b
        let cstr =
            fmt (ndigit+1) c
                |> Seq.rev |> String.Concat // reverse c to make addition easier
        let render = astr + bstr + cstr
        let dix = [| for c in render -> int64 (c - '0') |] // convert each character to its token index
        // x will be input to GPT and y will be the associated expected outputs
        let x = torch.tensor(dix[.. dix.Length-2], dtype=torch.long)
        let y = torch.tensor(dix[1 ..], dtype=torch.long) // predict the next token in the sequence
        y[Slice(stop=ndigit*2-1)] <- tensor -1 // we will only train in the output locations. -1 will mask loss to zero
        x, y

type AdderConfig =
    {
        seed : int
        data : AdditionDatasetConfig
        model : ModelConfig
        trainer : TrainerConfig
    } with
    
    static member get_config () =
        {
            seed = 3407
            data = AdditionDataset.get_default_config()
            model = { GPT.get_default_config() with model_type = "gpt-nano" }
            trainer = { Trainer.get_default_config() with learning_rate = 5e-4 } // the model we're using is so small that we can go a bit faster
        }

module Program =

    // get default config
    let config_ = AdderConfig.get_config ()
    printfn $"{config_}"
    set_seed config_.seed

    // construct train and test datasets
    let train_dataset = new AdditionDataset(config_.data, split="train")
    let test_dataset  = new AdditionDataset(config_.data, split="test")

    // construct the model
    let config =
        { config_ with
            model =
                { config_.model with
                    vocab_size = train_dataset.get_vocab_size()
                    block_size = train_dataset.get_block_size()
                } }
    let model = new GPT(config.model)

    // construct the trainer object
    let trainer = Trainer(config.trainer, model, train_dataset)

    // helper function for the evaluation of a model
    let eval_split (progress : TrainerProgress) split max_batches =
        let dataset = (Map ["train", train_dataset; "test", test_dataset])[split]
        let ndigit = config.data.ndigit
        let factors =
            torch.tensor(array2D [[for i in ndigit .. -1 .. 0 -> powi 10 i]])
                .``to``(progress.device)
        let loader = new MinDataLoader(dataset, batch_size=100, num_worker=0, drop_last=false)
        let results, _ =
            let dicts =
                max_batches
                    |> Option.map (fun max -> Seq.truncate max loader)
                    |> Option.defaultValue loader
            (([], 0), dicts)
                ||> Seq.fold (fun (results, mistakes_printed_already) (x, y) ->
                    let x = x.``to``(progress.device)
                    // isolate the first two digits of the input sequence alone
                    let d1d2 = x[Colon, Slice(stop=ndigit*2)]
                    // let the model sample the rest of the sequence
                    let d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=false) // using greedy argmax, not sampling
                    // isolate the last digit of the sampled sequence
                    let d3 = d1d2d3[Colon, Slice(-(ndigit+1))]
                    let d3 = d3.flip(1) // reverse the digits to their "normal" order
                    // decode the integers from individual digits
                    let d1i = (d1d2[Colon, Slice(stop=ndigit)] * factors[Colon, Slice(1)]).sum(1)
                    let d2i = (d1d2[Colon, Slice(ndigit, ndigit*2)] * factors[Colon, Slice(1)]).sum(1)
                    let d3i_pred = (d3 * factors).sum(1)
                    let d3i_gt = d1i + d2i // manually calculate the ground truth
                    // evaluate the correctness of the results in this batch
                    let correct = (torch.eq(d3i_pred, d3i_gt)).cpu() // Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
                    ((results, mistakes_printed_already), rangel(x.size(0)))
                        ||> Seq.fold (fun (results, mistakes_printed_already) i ->
                            let results = int(correct[i]) :: results
                            let mistakes_printed_already =
                                if (not (correct[i].item<bool>())) && mistakes_printed_already < 5 then // only print up to 5 mistakes to get a sense
                                    let get (t : Tensor) = t[i].item<int64>()
                                    printfn "GPT claims that %d + %d = %d but gt is %d" (get d1i) (get d2i) (get d3i_pred) (get d3i_gt)
                                    mistakes_printed_already + 1
                                else mistakes_printed_already
                            results, mistakes_printed_already))

        let results = Seq.toArray results
        let rt = torch.tensor(results, dtype=torch.float)
        printfn "%s final score: %.0f/%d = %.2f%% correct"
            split (rt.sum().item<float32>()) results.Length (100.0f * rt.mean().item<float32>())
        rt.sum()

    // iteration callback
    let mutable top_score = 0.0f
    let batch_end_callback progress =

        if progress.iter_num % 10 = 0 then
            printfn $"iter_dt {progress.iter_dt.TotalMilliseconds:f2}ms; iter {progress.iter_num}: train loss {progress.loss}"

        if progress.iter_num % 500 = 0 then
            // evaluate both the train and test score
            let train_max_batches =
                if config.data.ndigit > 2 then Some 5
                else Option.None // if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            let train_score, test_score =
                using (torch.no_grad()) (fun _ ->
                    eval_split progress "train" train_max_batches,
                    eval_split progress "test" Option.None)
            let score = (train_score + test_score).item<float32>()
            // save the model if this is the best score we've seen so far
            if score > top_score then
                top_score <- score
                printfn $"saving model with new top score of {score}"
                model.save("model.pt") |> ignore
            // revert model to training mode
            model.train()

    trainer.set_callback "on_batch_end" batch_end_callback

    // run the optimization
    trainer.run()
