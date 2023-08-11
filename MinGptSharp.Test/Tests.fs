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
