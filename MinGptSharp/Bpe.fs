﻿namespace MinGptSharp

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
        word
            |> Seq.pairwise
            |> Seq.distinct
            |> Seq.toArray

    open System.IO

    let get_encoder () =
        File.ReadLines "vocab.bpe"
            |> Seq.skip 1
            |> Seq.choose (fun merge_str ->
                match Seq.toList <| merge_str.Split(' ') with
                    | [first; second] -> Some (first, second)
                    | _ -> None)

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

    let bpe (token : string) =

        let rec loop (word : seq<string>) =
            let pairs = Bpe.get_pairs word
            let bigram =
                pairs
                    |> Seq.minBy (fun pair ->
                        bpe_ranks
                            |> Map.tryFind pair
                            |> Option.defaultValue Int32.MaxValue)
            if bpe_ranks.ContainsKey(bigram) then
                let new_word =
                    let pairs =
                        [|
                            yield! pairs
                            yield (snd pairs[pairs.Length - 1], "")
                        |]
                    (false, pairs)
                        ||> Seq.mapFold (fun merged (first, second) ->
                            if merged then
                                None, false
                            else
                                if (first, second) = bigram then
                                    Some (first + second), true
                                else
                                    Some first, false)
                        |> fst
                        |> Seq.choose id
                        |> Seq.toArray
                if new_word.Length > 1 then
                    loop new_word
                else Array.toSeq new_word
            else word

        if token.Length <= 1 then token
        else
            token
                |> Seq.map string
                |> loop
                |> String.concat " "

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
                let token_merged = bpe(token_translated).Split(' ')
                token_merged
        |]
