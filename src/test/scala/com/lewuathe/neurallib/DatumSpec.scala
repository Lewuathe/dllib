package com.lewuathe.neurallib

import breeze.linalg.DenseVector
import org.scalatest.FlatSpec

/**
 * Created by lewuathe on 5/12/15.
 */
class DatumSpec extends FlatSpec {
  "Channel" should "keep given data" in {
    val v = DenseVector[Double](1.0, 2.0, 3.0, 4.0)
    val c = new Channel(v, 2, 2)
    assert(c.getValues()(0) == 1.0)
    assert(c.getValues()(1) == 2.0)
    assert(c.getValues()(2) == 3.0)
    assert(c.getValues()(3) == 4.0)
  }
}
