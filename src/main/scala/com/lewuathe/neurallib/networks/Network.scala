package com.lewuathe.neurallib.networks

import com.lewuathe.neurallib.Datum
import com.lewuathe.neurallib.layers.Layer

abstract class Network {
  val layers: Seq[Layer]

  /**
   * Get all parameters of this network
   * @return
   */
  def params: Seq[Datum] = layers.map(_.param)

  def paramStructs: Seq[Layer.ParamStruct] = layers.map(_.paramStruct)

  def setParams(newParams: Seq[Datum]) = {
    for ((l, p) <- layers zip newParams) {
      l.param = p
    }
  }

  def updateParams(delta: Seq[Datum]) = {
    val newParams = for ((p, d) <- params zip delta) yield p + d
    setParams(newParams)
  }

  /**
   * Construct the zero parameter whose structure is same to own params.
   * @return
   */
  private[neurallib] def zeroParams: Seq[Datum] = layers.map(_.zeroParam)

  /**
   * Calculate total output of this network.
   * @param downstream
   * @return
   */
  def forward(downstream: Datum): Datum = {
    var d = downstream
    for (layer <- layers) {
      d = layer.forward(d)
    }
    d
  }

  /**
   * Calculate error difference of this network.
   * @param upstream
   * @return delta
   */
  def backward(upstream: Datum): Seq[Datum] = {
    var d = upstream
    var ret = Seq[Datum]()
    for (layer <- layers) {
      ret = layer.delta(d) match {
        case Some(delta) if layer.isParamLayer => delta +: ret
        case None => ret
      }
      d = layer.backward(d)
    }
    ret
  }

  /**
   * Do backpropagation through network.
   * @param input
   * @return
   */
  def delta(input: Datum): Seq[Datum]
    = backward(forward(input))
}

object Network {

}