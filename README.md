dllib 
================

This is deep learning module running on Apache Spark. Powered by Scala programming language.

# How to use

Write below configuration in your build.sbt.

## Example

```scala
object Main {
  val nn = NN3(Array(784, 100, 10), 0.1, (iteration: Int, nn3: NN) => {
    var correctNum = 0
    for (i <- 0 until testxs.rows) {
      val ans = nn3.predict(testxs(i, ::).t)
      if (testAnswer(ans, testys(i, ::).t)) correctNum += 1
    }
    val accuracy = (correctNum * 100.0) / testDataCount
    println(f"#$iteration%02d : ${correctNum}/${testDataCount}, ${accuracy}")
  })
  nn.tied = false
  nn.train(xs, ys)
}
```


# License

[Apache v2](http://www.apache.org/licenses/LICENSE-2.0)

# Author

* Kai Sasaki([@Lewuathe](https://github.com/Lewuathe))
