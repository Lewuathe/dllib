package com.lewuathe.dllib

import com.lewuathe.dllib.example.XORApp
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest._

class XORSpec extends FlatSpec with Matchers {
  "NN3" should "learn XOR behaviour" in {
    val conf = new SparkConf()
    val sc = new SparkContext(master = "local[*]",
      appName = "XORSpec", conf = conf)
    sc.setLogLevel("WARN")
    val app = XORApp
    app.numIterations = 2000
    app.learningRate = 0.7
    app.submit(sc)
  }
}
