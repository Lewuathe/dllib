package com.lewuathe.neurallib

import com.lewuathe.neurallib.layers.SquaredLossLayer
import org.scalatest.FlatSpec

/**
 * Created by kaisasak on 6/1/15.
 */
class SquaredLossLayerSpec extends FlatSpec {
  "SquaredLossLayer" should "return proper error" in {
    val squaredLossLayer = SquaredLossLayer()
    val input = DatumUtil.createLabelDatum()
    val label = DatumUtil.createReversedLabelDatum()

    val loss = squaredLossLayer.loss(input, label)
    assert(loss.getData.length == 1)
    assert(loss.getData(0).getValues(0) == 3.0)
    assert(loss.getData(0).getValues(1) == 1.0)
    assert(loss.getData(0).getValues(2) == 1.0)
    assert(loss.getData(0).getValues(3) == 3.0)
  }
}
