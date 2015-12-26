package com.lewuathe.dllib

import org.scalatest._

import breeze.linalg.Vector

class LabelEncoderSpec extends FlatSpec with Matchers {
  "Integer" should "be encoded into one hot vector" in {
    val label = 3
    val labelCount = 10
    val v = util.encodeLabel(3, labelCount)
    v(label) should be (1.0)
  }

  "Vector" should "be decoded into one max integer" in {
    val v = Vector[Double](0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.4, 0.1, 0.1)
    val label = util.decodeLabel(v)
    label should be (7.0)
  }
}
