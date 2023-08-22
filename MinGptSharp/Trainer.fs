namespace MinGptSharp

(*
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
*)

open System
open System.Collections.Generic

open TorchSharp
open type torch
open type utils.data

#nowarn "25"   // allow pattern matching on listss

type TrainerProgress =
    {
        iter_num : int
        iter_dt : TimeSpan
        loss : float32
        device : string
    }

type Trainer(config : TrainerConfig, model : GPT, train_dataset : MinDataset) =

    let callbacks = Dictionary<string, ResizeArray<TrainerProgress -> unit>>()

    // determine the device we'll train on
    let device =
        if config.device = "auto" then
            if torch.cuda.is_available() then "cuda" else "cpu"
        else
            config.device
    let model = model.``to``(device)
    do printfn $"running on device {device}"

    let trigger_callbacks (onevent: string) iter_num =
        let list =
            match callbacks.TryGetValue(onevent) with
                | true, list -> list :> seq<_>
                | false, _ -> Seq.empty
        for callback in list do
            callback iter_num

    static member get_default_config() =
        {
            // device to train on
            device = "auto"
            // dataloder parameters
            num_workers = 4
            // optimizer parameters
            max_iters = -1
            batch_size = 64
            learning_rate = 3e-4
            betas = (0.9, 0.95)
            weight_decay = 0.1 // only applied on matmul weights
            grad_norm_clip = 1.0
        }

    member _.Device = device

    member _.set_callback onevent callback =
        callbacks[onevent] <- ResizeArray [callback]

    member _.add_callback(onevent) callback =
        callbacks[onevent].Add(callback)

    member _.run() =

        // setup the optimizer
        let optimizer = model.configure_optimizers(config)

        // setup the dataloader
        let train_loader =
            (*
            new DataLoader(
                train_dataset,
                sampler=torch.utils.data.RandomSampler(train_dataset, replacement=true, num_samples=int(1e10)),
                shuffle=false,
                pin_memory=true,
                batch_size=config.batch_size,
                num_workers=config.num_workers)
            *)
            new MinDataLoader(train_dataset, config.batch_size, shuffle=true, num_worker=config.num_workers)

        model.train()

        let rec loop iter_num iter_time (data_iter : IEnumerator<_>) =

            if data_iter.MoveNext() then

                let iter_time =
                    use _scope = torch.NewDisposeScope()

                    // fetch the next batch (x, y)
                    let (x : Tensor), (y : Tensor) = data_iter.Current
                    let x = x.``to``(device)
                    let y = y.``to``(device)

                    // forward the model
                    let _logits, loss = model.forward(x, y)

                    // backprop and update the parameters
                    optimizer.zero_grad((*set_to_none=true*))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip) |> ignore
                    optimizer.step() |> ignore

                    let tnow = DateTime.Now
                    let iter_dt = tnow - iter_time
                    trigger_callbacks "on_batch_end"
                        {
                            iter_num = iter_num
                            iter_dt = iter_dt
                            loss = loss.item<float32>()
                            device = device
                        }
                    tnow

                // termination conditions
                if config.max_iters <= 0 || iter_num < config.max_iters then
                    loop (iter_num + 1) iter_time data_iter

            else
                train_loader.GetEnumerator() |> loop (iter_num + 1) iter_time

        train_loader.GetEnumerator() |> loop 0 DateTime.Now
