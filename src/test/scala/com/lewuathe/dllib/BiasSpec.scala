package com.lewuathe.dllib

import org.scalatest._

class BiasSpec extends FlatSpec with Matchers {
  "Bias ID's length" should "be equal to 16" in {
    val b = Bias(4)
    b.id.length should be (16)
  }
  "Bias" should "match size" in {
    val b = Bias(4)
    b.value.size should be (4)
  }
}
