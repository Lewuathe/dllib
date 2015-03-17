neurallib [![Build Status](https://travis-ci.org/Lewuathe/neurallib.svg?branch=master)](https://travis-ci.org/Lewuathe/neurallib)
================

This is deep learning module running on JVM. Powered by Scala programming language.

# How to use

[Maven Central](http://mvnrepository.com/artifact/com.lewuathe/neurallib_2.10/0.0.1)

Write below configuration in your build.sbt.

```
libraryDependencies += "com.lewuathe" % "neurallib_2.10" % "0.0.2"
```

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

[MIT License](http://opensource.org/licenses/MIT)

# Author

* Kai Sasaki([@Lewuathe](https://github.com/Lewuathe))
* Shuichi Suzuki([@shue116](https://github.com/shoe116))
