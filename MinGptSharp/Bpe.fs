namespace MinGptSharp

(*
A byte-pair encoder translates a string into a sequence of tokens,
where each token is an integer representing a frequently-occurring
series of characters in English (e.g. "ing"). The encoding process is:
  
1. Use a regex to break the input text into pieces, such as
    " Karpathy". (Note the leading space in this example.)
  
2. Encode each byte of each piece into a Unicode character. E.g.
    " Karpathy" -> "ĠKarpathy". (The leading space is encoded as
    a visible Unicode character.)
  
3. Merge the Unicode characters of each piece into tokens. E.g.
    "ĠKarpathy" -> "ĠK", "arp", "athy"
  
4. Encode each token as an integer. E.g. "ĠK", "arp", "athy" ->
    509, 5117, 10036.

In this file,  I've attempted to follow minGPT's general structure,
but with a functional approach and my own comments.
 *)

module Bpe =

    /// Maps bytes to Unicode characters.
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

    /// Breaks the given sequence into distinct pairs of adjacent items.
    /// E.g. "h", "el", "lo", "!" -> ("h", "el"), ("el", "lo"), ("lo", "!")
    let get_pairs word =
        word
            |> Seq.pairwise
            |> Seq.distinct
            |> Seq.toArray

open System
open System.Text

/// A byte-pair encoder/decoder.
type Encoder(encoder, bpe_merges : seq<string * string>) =

    // byte encoder/decoder
    let byte_encoder = Bpe.bytesToUnicode
    let byte_decoder = Map [ for KeyValue(k, v) in byte_encoder do v, k ]

    // bpe token encoder/decoder
    let encoder : Map<_, _> = encoder
    let decoder = Map [ for KeyValue(k, v) in encoder do v, k ]

    /// Defines the order in which pieces of text are to be merged.
    let bpe_ranks =
        bpe_merges
            |> Seq.mapi (fun i x -> x, i)
            |> Map

    /// String-breaking regex.
    let pat =
        """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            |> RegularExpressions.Regex

    /// Merges characters in the given string.
    /// E.g. "ĠKarpathy" -> "ĠK arp athy".
    let bpeRaw (token : string) =

        /// Merges a pair of pieces of the given word.
        let rec merge (word : string[]) =
            if word.Length < 2 then word
            else
                    // find the lowest-rank bigram that can be merged
                let pairs = Bpe.get_pairs word
                let bigram =
                    pairs
                        |> Seq.minBy (fun pair ->
                            bpe_ranks
                                |> Map.tryFind pair
                                |> Option.defaultValue Int32.MaxValue)

                    // merge all occurrences of the bigram
                if bpe_ranks.ContainsKey(bigram) then
                    let pairs =
                        seq {
                            yield! pairs
                            yield (Array.last pairs |> snd, "")   // add pair at the end for the last element
                        }
                    (false, pairs)
                        ||> Seq.mapFold (fun merged (first, second) ->
                            if merged then
                                None, false                       // ignore this pair because previous pair was merged
                            elif (first, second) = bigram then
                                Some (first + second), true       // merge this pair
                            else
                                Some first, false)
                        |> fst
                        |> Seq.choose id
                        |> Seq.toArray
                        |> merge
                else word

        assert(token.Contains(' ') |> not)
        token.ToCharArray()
            |> Array.map string    // convert each character to a string of lenth 1
            |> merge               // merge all known bigrams
            |> String.concat " "   // flatten using space character as delimiter

        // memoize for speed
    let bpe = memoize bpeRaw

    do
        assert(byte_decoder.Count = byte_encoder.Count)
        assert(decoder.Count = encoder.Count)

    /// Encodes the given text as a sequence of integers.
    member _.Encode(text) =
        pat.Matches(text)
            |> Seq.collect (fun mtch ->
                let token_bytes = Encoding.UTF8.GetBytes(mtch.Value)
                let token_translated =
                    String [|
                        for b in token_bytes do
                            byte_encoder[int b]
                    |]
                let token_merged = bpe(token_translated).Split(' ')
                [| for bpe_token in token_merged -> encoder[bpe_token] |])
            |> Seq.toArray

    /// Decodes the given sequence of integers to text.
    member _.Decode(bpe_idx) =
        let tokens_merged = [| for token in bpe_idx -> decoder[token] |]
        let tokens_flat = String.concat "" tokens_merged
        let tokens_bytes = [| for c in tokens_flat -> byte byte_decoder[c] |]
        Encoding.UTF8.GetString(tokens_bytes)

module Encoder =

    open System.IO
    open System.Text.Json

    /// Creates a GPT BPE encoder/decoder.
    let get_encoder () =

            // load mappings from token to integer
        let encoder =
            use reader = new StreamReader("encoder.json")
            JsonSerializer.Deserialize<Map<string, int>>(reader.BaseStream)
        assert(encoder.Count = 256 (*byte tokens*) + 50000 (*merged tokens*) + 1 (*end-of-text token*))

            // load the tree structure that indicates how to merge
            // characters into tokens
        let bpe_merges =
            File.ReadLines "vocab.bpe"
                |> Seq.skip 1   // skip version #
                |> Seq.choose (fun merge_str ->
                    match Seq.toList <| merge_str.Split(' ') with
                        | [first; second] -> Some (first, second)
                        | [] -> None
                        | _ -> failwith $"Unexpected line: {merge_str}")
                |> Seq.toArray
        assert(bpe_merges.Length = 50000)

        Encoder(encoder, bpe_merges)
