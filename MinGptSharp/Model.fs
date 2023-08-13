namespace MinGptSharp

open System

open TorchSharp
open type TorchSharp.torch

type NewGELU() =
    inherit nn.Module<Tensor, Tensor>("NewGELU")
    let s (x : float) = x.ToScalar()
    override _.forward(x) =
        s 0.5 * x * (s 1.0 + torch.tanh(s (Math.Sqrt(2.0 / Math.PI)) * (x + s 0.044715 * torch.pow(x, s 3.0))))
