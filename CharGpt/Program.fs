namespace MinGptSharp

(*
Trains a character-level language model.
*)

open System

open TorchSharp
open type torch
open type utils.data
open type TensorIndex
open FSharp.Core.Operators   // reclaim "int64" and other F# operators

type CharDatasetConfig =
    {
        block_size : int
    }

/// Emits batches of characters
type CharDataset(config, data : string) =
    inherit MinDataset()

    let chars = set data
    let data_size, vocab_size_ = data.Length, chars.Count
    do printfn "data has %d characters, %d unique." data_size vocab_size_

    let stoi = Map [ for i, ch in Seq.indexed chars -> ch, i ]
    let itos = Map [ for i, ch in Seq.indexed chars -> i, ch ]

    static member get_default_config() =
        {
            block_size = 128
        }

    member _.Itos(i) = itos[i]
    member _.Stoi(ch) = stoi[ch]

    member _.get_vocab_size() =
        vocab_size_

    member _.get_block_size() =
        config.block_size

    override _.Count with get() =
        int64 (data.Length - config.block_size)

    override _.GetTensor(idx) =
        // grab a chunk of (block_size + 1) characters from the data
        let chunk = data[int idx .. int idx + config.block_size]
        assert(chunk.Length = config.block_size + 1)
        // encode every character to an integer
        let dix = [| for ch in chunk -> stoi[ch] |]
        // return as tensors
        let x = torch.tensor(dix[.. dix.Length-2], dtype=torch.long)
        let y = torch.tensor(dix[1 ..], dtype=torch.long)
        x, y

type CharConfig =
    {
        seed : int
        data : CharDatasetConfig
        model : ModelConfig
        trainer : TrainerConfig
    } with
    
    static member get_config () =
        {
            seed = 3407
            data = CharDataset.get_default_config()
            model = { GPT.get_default_config() with model_type = "gpt-mini" }
            trainer = { Trainer.get_default_config() with learning_rate = 5e-4 } // the model we're using is so small that we can go a bit faster
        }

module Program =

    // get default config
    let config_ = CharConfig.get_config ()
    printfn $"{config_}"
    set_seed config_.seed

    // construct the training dataset
    let text = System.IO.File.ReadAllText("Input.txt")
    let train_dataset = new CharDataset(config_.data, text)

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

    // iteration callback
    let batch_end_callback progress =

        if progress.iter_num % 10 = 0 then
            printfn $"iter_dt {progress.iter_dt.TotalMilliseconds:f2}ms; iter {progress.iter_num}: train loss {progress.loss}"

        if progress.iter_num % 500 = 0 then
            model.eval()
            using (torch.no_grad()) (fun _ ->
                // sample from the model...
                let context = "O God, O God!"
                let x = torch.tensor([| for ch in context -> train_dataset.Stoi(ch) |], dtype=torch.long)
                let x = x[None, Ellipsis].``to``(trainer.Device)
                let y = model.generate(x, 500, temperature=1.0, do_sample=true, top_k=10)[0]
                let completion = String ([| for i in y.data<int64>() -> train_dataset.Itos(int i) |])
                printfn "%s" completion)
            model.save("model.pt") |> ignore
            // revert model to training mode
            model.train()

    trainer.set_callback "on_batch_end" batch_end_callback

    // run the optimization
    trainer.run()
