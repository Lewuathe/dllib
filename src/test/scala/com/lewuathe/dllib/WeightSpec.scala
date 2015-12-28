package com.lewuathe.dllib

import org.scalatest._

class WeightSpec extends FlatSpec with Matchers {
  "Weight ID's length" should "be equals to 16" in {
    val w = Weight(3, 2)
    w.id.length should be (16)
  }
  "Weight" should "match column and rows" in {
    val w = Weight(3, 2)
    w.value.rows should be (3)
    w.value.cols should be (2)
  }
}
