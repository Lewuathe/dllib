package com.lewuathe.neurallib

import com.lewuathe.neurallib.layers.SigmoidLayer
import org.scalatest.FlatSpec

/**
 * Created by sasakikai on 5/26/15.
 */
class SigmoidLayerSpec extends FlatSpec {
  "SigmoidLayer" should "calculate correctly" in {
    val input = DatumUtil.createDatum()
    val sigmoidLayer = SigmoidLayer()
    val ret = sigmoidLayer.forward(input)
    for (c <- ret.getData) {
      assert(0.70 < c.getValues(0) && c.getValues(0) < 0.75)
      assert(0.85 < c.getValues(1) && c.getValues(1) < 0.90)
      assert(0.95 < c.getValues(2) && c.getValues(2) < 1.0)
    }
  }

}
