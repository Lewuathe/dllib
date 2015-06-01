package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.{Channel, Datum}

/**
 * Created by lewuathe on 6/1/15.
 */
class SquaredLossLayer extends LossLayer {
  /**
   * Calculate loss based on input data and label
   * @param data
   * @param label
   * @return
   */
  override def loss(data: Datum, label: Datum): Datum = {
    require(label.channels == 1)
    require(data.channels == label.channels)

    val labelVector = label.getData(0).getValues
    val dataVector = data.getData(0).getValues

    require(labelVector.length == dataVector.length)

    val lossChannel
      = new Channel(((labelVector - dataVector) :* (labelVector - dataVector)).map(Math.sqrt),
                     data.height, data.width)

    Datum(Seq(lossChannel), 1, data.height, data.width)
  }

  override def isParamLayer: Boolean = false
}

object SquaredLossLayer {
  def apply(): SquaredLossLayer = new SquaredLossLayer
}
