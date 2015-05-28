package com.lewuathe.neurallib

import com.lewuathe.neurallib.layers.InnerProductLayer
import org.scalatest.FlatSpec

/**
 * Created by sasakikai on 5/26/15.
 */
class InnerProductLayerSpec extends FlatSpec {
  val inputDim = DatumUtil.datumHeight * DatumUtil.datumWidth
  val outputDim = 3

  "InnerProductLayer" should "calculate correctly" in {
    val conf = Map(
      "channels" -> DatumUtil.datumChannels.toString,
      "input" -> inputDim.toString,
      "output" -> outputDim.toString
    )
    val innerProductLayer = InnerProductLayer(conf)
    assert(innerProductLayer.inputDim == 4)
    val input = DatumUtil.createDatum()
    val ret = innerProductLayer.forward(input)
    assert(ret.getData.length == 3)
    for (c <- ret.getData) {
      assert(-10.0 < c.getValues(0) && c.getValues(0) < 10.0)
    }
  }

  "InnerProductLayer" should "return correct size in backward" in {
    val conf = Map(
      "channels" -> DatumUtil.datumChannels.toString,
      "input" -> inputDim.toString,
      "output" -> outputDim.toString
    )

    val innerProductLayer = InnerProductLayer(conf)
    val input = DatumUtil.createDatum()
    val ret = innerProductLayer.forward(input)
    assert(ret.getData.length == 3)

    val back = innerProductLayer.backward(ret)
    assert(back.getData.length == 3)
    for (c <- back.getData) {
      assert(c.getValues.length == 4)
    }
  }

  "InnerProductLayer" should "return correct size delta of own parameter" in {
    val conf = Map(
      "channels" -> DatumUtil.datumChannels.toString,
      "input" -> inputDim.toString,
      "output" -> outputDim.toString
    )

    val innerProductLayer = InnerProductLayer(conf)
    val input = DatumUtil.createDatum()
    val ret = innerProductLayer.forward(input)
    assert(ret.getData.length == 3)

    val delta = innerProductLayer.delta(ret).get
    assert(delta.getData.length == 3)
    for (c <- delta.getData) {
      assert(c.getValues.length == inputDim * outputDim)
    }
  }

  "InnerProductLayer" should "return Matrix shaped parameter" in {
    val conf = Map(
      "channels" -> DatumUtil.datumChannels.toString,
      "input" -> inputDim.toString,
      "output" -> outputDim.toString
    )

    val innerProductLayer = InnerProductLayer(conf)
    val matrixParam = innerProductLayer.matrixParam
    for (m <- matrixParam) {
      assert(m.rows == 3)
      assert(m.cols == 4)
    }
  }
}
