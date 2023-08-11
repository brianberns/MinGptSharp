namespace MinGptSharp

[<AutoOpen>]
module Utils =

    let range n = seq { 0 .. n - 1 }

module Bpe =

    let bytesToUnicode =

            // integers that render fine in their original form
        let bs =
            set [
                yield! [int '!' .. int '~']
                yield! [int '¡' .. int '¬']
                yield! [int '®' .. int 'ÿ']
            ]
        assert(bs.Count = 188)

            // shift other integers as necessary
        let size = 1 <<< 8
        (0, range size)
            ||> Seq.mapFold (fun n b ->
                let c, n' =
                    if bs.Contains(b) then b, n
                    else size + n, n + 1   // map to the next available "nice" character
                (b, char c), n')
            |> fst
            |> Map
