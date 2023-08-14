namespace MinGptSharp

open Microsoft.VisualStudio.TestTools.UnitTesting

type Assert() =

    static member AreEqual(expected : 't, actual : 't) =
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.AreEqual<'t>(expected, actual)

    static member AreEqual(expected : 't, actual : 't, message : string) =
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.AreEqual<'t>(expected, actual, message)

[<TestClass>]
type Bpe() =

    [<TestMethod>]
    member _.BytesToUnicode() =
        let map = Bpe.bytesToUnicode
        Assert.AreEqual(256, map.Count)
        Assert.AreEqual('!', map[int '!'])
        Assert.AreEqual('Ā', map[0])
        Assert.AreEqual('Ġ', map[int ' '])

    [<TestMethod>]
    member _.Encode() =

        let text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
        let encoder = Encoder.get_encoder ()

            // encode
        let expected =
            [| 15496; 3228; 314; 1101; 10948; 73; 509; 5117; 10036; 13; 632; 338; 33160; 13; 266; 405; 83; 1058; 35; 12520; 97; 245 |]
        let actual = encoder.Encode(text)
        for expectedItem, actualItem in Array.zip expected actual do
            Assert.AreEqual(expectedItem, actualItem)

            // decode
        Assert.AreEqual(text, encoder.Decode(actual))
