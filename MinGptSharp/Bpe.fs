namespace MinGptSharp

[<AutoOpen>]
module Utils =

    let range n = seq { 0 .. n - 1 }

open System
open System.Text

/// Byte pair encoder
module Bpe =

    let bytesToUnicode =

            // characters that render fine in their original form
        let bs =
            set [
                yield! [int '!' .. int '~']
                yield! [int '¡' .. int '¬']
                yield! [int '®' .. int 'ÿ']
            ]

            // shift other characters as necessary
        let size = 1 <<< 8
        (0, range size)
            ||> Seq.mapFold (fun n b ->
                let c, n' =
                    if bs.Contains(b) then b, n
                    else size + n, n + 1   // map to the next available "nice" character
                (b, char c), n')
            |> fst
            |> Map

    let get_pairs word =
        Seq.pairwise word |> set

type Encoder(encoder, bpe_merges : seq<string * string>) =

    // byte encoder/decoder
    let byte_encoder = Bpe.bytesToUnicode
    let byte_decoder = Map [ for KeyValue(k, v) in byte_encoder do v, k ]

    // bpe token encoder/decoder
    let encoder : Map<_, _> = encoder
    let decoder = Map [ for KeyValue(k, v) in encoder do v, k ]

    let bpe_ranks =
        bpe_merges
            |> Seq.mapi (fun i x -> x, i)
            |> Map

    let pat =
        """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            |> RegularExpressions.Regex

    (*
    let bpe token =
        let word = token
        let pairs = Bpe.get_pairs word

        let rec loop pairs =
            let bigram =
                pairs
                    |> Seq.minBy (fun pair ->
                        bpe_ranks
                            |> Map.tryFind pair
                            |> Option.defaultValue Int32.MaxValue)
            if bpe_ranks.ContainsKey(pair) then
                let first, second = bigram

                let rec loop i =
                    let j =
                        word[i..]
                            |> Seq.tryFindIndex ((=) first)
                            |> Option.map (fun j -> word[i..j-1])
                                loop j
    *)

    do
        assert(byte_decoder.Count = byte_encoder.Count)
        assert(decoder.Count = encoder.Count)

    member _.Tokenize(text) =
        pat.Matches(text)
            |> Seq.map (fun mtch -> mtch.Value)
            |> Seq.toArray

    member this.Encode(text) =
        let tokens = this.Tokenize(text)
        [|
            for token in tokens do
                let token_bytes = Encoding.UTF8.GetBytes(token)
                let token_translated =
                    String [|
                        for b in token_bytes do
                            byte_encoder[int b]
                    |]
                token_translated
        |]