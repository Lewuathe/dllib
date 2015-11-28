package com.lewuathe.dllib

import org.scalatest._

class WeightSpec extends FlatSpec with Matchers {
  "Weight ID" should "be equals to 16" in {
    val w = Weight(3, 2)
    w.id.length should be (16)
  }
  "Weight" should "match column and rows" in {
    val w = Weight(3, 2)
    w.value.numRows should be (3)
    w.value.numCols should be (2)
  }
}
