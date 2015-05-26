package com.lewuathe.neurallib

import com.lewuathe.neurallib.layers.InnerProductLayer
import org.scalatest.FlatSpec

/**
 * Created by sasakikai on 5/26/15.
 */
class InnerProductLayerSpec extends FlatSpec {
  "InnerProductLayer" should "calculate correctly" in {
    val conf = Map(
      "channels" -> DatumUtil.datumChannels.toString,
      "input" -> (DatumUtil.datumHeight * DatumUtil.datumWidth).toString,
      "output" -> "3"
    )
    val innerProductLayer = InnerProductLayer(conf)
    val input = DatumUtil.createDatum()
    val ret = innerProductLayer.forward(input)
    assert(ret.getData.length == 3)
    for (c <- ret.getData) {
      assert(-10.0 < c.getValues(0) && c.getValues(0) < 10.0)
    }
  }
}
