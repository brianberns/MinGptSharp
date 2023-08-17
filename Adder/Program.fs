﻿namespace MinGptSharp

(*
Trains a GPT to add n-digit numbers.
*)

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
type AdditionDataset(config, split (*train/test*)) as self =
    inherit Dataset()

    let pow x y =
        Seq.replicate y x |> Seq.reduce (*)

    // split up all addition problems into either training data or test data
    let ndigit = config.ndigit
    do assert(ndigit <= 3) // "the lines below would be very memory inefficient, in future maybe refactor to support"
    let num = pow (pow 10 ndigit) 2 // total number of possible addition problems with ndigit numbers
    let rng = new torch.Generator()
    do rng.manual_seed(1337) |> ignore
    let perm = torch.randperm(num, generator=rng)
    let num_test = min ((num / 5)) 500 // 20% of the whole dataset, or only up to 500
    let ixes =
        if split = "test" then perm[Slice(stop=num_test)]
        else perm[Slice(num_test)]

    let rev (str : string) =
        str |> Seq.rev |> System.String.Concat

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
        3 * config.ndigit + 1 - 1

    override _.Count with get() = ixes.NumberOfElements

    override _.GetTensor(idx) =
        // given a problem index idx, first recover the associated a + b
        let idx = ixes[idx].item()
        let nd = pow 10 ndigit
        let a = idx // nd
        let b = idx %  nd
        // calculate the "label" of the addition problem a + b
        let c = a + b
        // encode the digits of a, b, c into strings
        let fmt n = Printf.StringFormat<int -> string>(sprintf "%0d" n)
        let astr = sprintf (fmt ndigit) a
        let bstr = sprintf (fmt ndigit) b
        let cstr = sprintf (fmt (ndigit+1)) c |> rev // reverse c to make addition easier
        let render = astr + bstr + cstr
        let dix = [| for c in render -> int64 c |] // convert each character to its token index
        // x will be input to GPT and y will be the associated expected outputs
        let x = torch.tensor(dix[.. dix.Length-2], dtype=torch.long)
        let y = torch.tensor(dix[1 ..], dtype=torch.long) // predict the next token in the sequence
        let stop = ndigit*2-1
        let slice = Slice(stop=stop)
        y[slice] <- (-1).ToTensor() // we will only train in the output locations. -1 will mask loss to zero
        dict [ "x", x; "y", y ] |> System.Collections.Generic.Dictionary

type AdderConfig =
    {
        // system
        seed : int
        work_dir : string

        // data
        data : AdditionDatasetConfig

        // model
        model : ModelConfig
        model_type : string

        // trainer
        trainer : TrainerConfig
        learning_rate : float
    } with
    
    static member get_config () =
        {
            seed = 3407
            work_dir = "./out/adder"

            data = AdditionDataset.get_default_config()

            model = GPT.get_default_config()
            model_type = "gpt-nano"

            trainer = Trainer.get_default_config()
            learning_rate = 5e-4 // the model we"re using is so small that we can go a bit faster
        }

(*

if __name__ == "__main__":

    // get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    // construct train and test datasets
    train_dataset = AdditionDataset(config.data, split="train")
    test_dataset  = AdditionDataset(config.data, split="test")

    // construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    // construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    // helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {"train":train_dataset, "test":test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            // isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            // let the model sample the rest of the sequence
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) // using greedy argmax, not sampling
            // isolate the last digit of the sampled sequence
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) // reverse the digits to their "normal" order
            // decode the integers from individual digits
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i // manually calculate the ground truth
            // evaluate the correctness of the results in this batch
            correct = (d3i_pred == d3i_gt).cpu() // Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: // only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    // iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            // evaluate both the train and test score
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] // if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, "train", max_batches=train_max_batches)
                test_score  = eval_split(trainer, "test",  max_batches=None)
            score = train_score + test_score
            // save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            // revert model to training mode
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)

    // run the optimization
    trainer.run()
*)
