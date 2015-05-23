package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.{Channel, DiffChannel, Datum, sigmoid}
import breeze.linalg.Vector

/**
 * Created by lewuathe on 5/23/15.
 */
class SigmoidLayer extends Layer {
  /**
   * Initialize parameters of this layer
   */
  override def init(conf: Map[String, String]): Unit = {
    ;
  }

  /**
   * Calculate layer output according
   * to concrete implementation.
   * @param downstream
   * @return output Datum
   */
  override def forward(downstream: Datum): Datum = {
    prevForward = downstream.map(sigmoid)
    prevForward
  }

  /**
   * Calculate error update difference on each input error
   * for weights and biases
   * @param upstream
   * @return parameter delta
   */
  override def delta(upstream: Datum): Option[Datum] = None


  override def isParamLayer: Boolean = false

  /**
   * Calculate layer output with error differentiation
   * according to concrete implementation. Given input
   * @param upstream
   */
  override def backward(upstream: Datum): Datum = {
    val data = upstream.getData
    require(data.length > 0)
    require(data(0).isInstanceOf[DiffChannel])

    val previous = prevForward.getData
    val newChannels = for ((ch, prev)
                           <- data zip prevForward.getData) yield {
      // upstream diff
      val diffs = ch.getValues.toArray
      // data which calculated when feed forwarding
      val prevs = prev.getValues.toArray

      val newDiffs = Vector(
        (prevs zip diffs).map({ case (p, d) => p * (1.0 - p) * d }).toArray
      )
      new Channel(newDiffs, ch.getHeight, ch.getWidth)
    }

    // Sigmoid layer should not change the format of datum
    require(newChannels.length == data.length)
    Datum(newChannels, upstream.channels, upstream.height, upstream.width)
  }
}
