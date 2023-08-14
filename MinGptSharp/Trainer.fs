namespace MinGptSharp

(*
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
*)

open TorchSharp
open type TorchSharp.torch
open type TorchSharp.torch.utils.data

type TrainerConfig =
    {
        device : string
        num_workers : int
        max_iters : int
        batch_size : int
        learning_rate : float
        betas : float * float
        weight_decay : float
        grad_norm_clip : float
    }

type Trainer(config, model : GPT, train_dataset) as self =

    static let get_default_config () =
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

    let config = config
    let optimizer = None
    let train_dataset = train_dataset
    let callbacks = System.Collections.Generic.Dictionary<string, ResizeArray<Trainer -> unit>>()

    // determine the device we'll train on
    let device =
        if config.device = "auto" then
            if torch.cuda.is_available() then "cuda" else "cpu"
        else
            config.device
    let model = model.``to``(device)
    do printfn $"running on device {device}"

    // variables that will be assigned to trainer class later for logging and etc
    let iter_num = 0
    let iter_time = 0.0
    let iter_dt = 0.0

    let add_callback (onevent: string) callback =
        callbacks[onevent].Add(callback)

    let set_callback (onevent: string) callback =
        callbacks[onevent] = ResizeArray [callback]

    let trigger_callbacks (onevent: string) =
        let list =
            match callbacks.TryGetValue(onevent) with
                | true, list -> list :> seq<_>
                | false, _ -> Seq.empty
        for callback in list do
            callback(self)

    let run() =

        // setup the optimizer
        let optimizer = model.configure_optimizers(config)

        // setup the dataloader
        let train_loader =
            new DataLoader(
                train_dataset,
                sampler=torch.utils.data.RandomSampler(train_dataset, replacement=true, num_samples=int(1e10)),
                shuffle=false,
                pin_memory=true,
                batch_size=config.batch_size,
                num_workers=config.num_workers)

        model.train()
        let iter_num = 0
        let iter_time = System.DateTime.Now
        let data_iter = iter(train_loader)

        let rec loop () =

            // fetch the next batch (x, y) and re-init iterator if needed
            let batch =
                try
                    next(data_iter)
                with StopIteration ->
                    let data_iter = iter(train_loader)
                    next(data_iter)
            let batch = [for t in batch -> t.``to``(device)]
            let x, y = batch

            // forward the model
            let logits, loss = model.forward(x, y)

            // backprop and update the parameters
            model.zero_grad(set_to_none=true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip) |> ignore
            optimizer.step()

            trigger_callbacks("on_batch_end")
            iter_num += 1
            let tnow = System.DateTime.Now
            let iter_dt = tnow - iter_time
            let iter_time = tnow

            // termination conditions
            if config.max_iters <= 0 || iter_num < config.max_iters then
                loop ()

        loop ()
