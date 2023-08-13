namespace MinGptSharp

open Microsoft.VisualStudio.TestTools.UnitTesting

type Assert() =

    static member AreEqual(expected : 't, actual : 't) =
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.AreEqual<'t>(expected, actual)

    static member AreEqual(expected : 't, actual : 't, message : string) =
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.AreEqual<'t>(expected, actual, message)

[<TestClass>]
type TestClass () =

    [<TestMethod>]
    member _.Bpe() =
        let map = Bpe.bytesToUnicode
        Assert.AreEqual(256, map.Count)
        Assert.AreEqual('!', map[int '!'])
        Assert.AreEqual('Ā', map[0])
        Assert.AreEqual('Ġ', map[int ' '])

    [<TestMethod>]
    member _.Tokenize() =
        let text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
        let pairs =
            Array.zip
                (Encoder(Map.empty, Seq.empty).Tokenize(text))
                [| "Hello"; "!!"; " I"; "'m"; " Andrej"; " Karpathy"; "."; " It"; "'s"; " 2022"; "."; " w"; "00"; "t"; " :"; "D"; " 🤗" |]
        for expected, actual in pairs do
            Assert.AreEqual(expected, actual)

    [<TestMethod>]
    member _.Encode() =

        let text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
        let expected =
            [|
                [| "Hello" |]
                [| "!!" |]
                [| "ĠI" |]
                [| "'m" |]
                [| "ĠAndre"; "j" |]
                [| "ĠK"; "arp"; "athy" |]
                [| "." |]
                [| "ĠIt" |]
                [| "'s" |]
                [| "Ġ2022" |]
                [| "." |]
                [| "Ġw" |]
                [| "00" |]
                [| "t" |]
                [| "Ġ:" |]
                [| "D" |]
                [| "ĠðŁ"; "¤"; "Ĺ" |]
            |]

        let actual =
            let encoder = Encoder.get_encoder ()
            encoder.Encode(text)

        for expecteds, actuals in Array.zip expected actual do
            for expectedStr, actualStr in Array.zip expecteds actuals do
                Assert.AreEqual(expectedStr, actualStr)
