package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.Datum

/**
 * Created by lewuathe on 5/23/15.
 */
class InnerProductLayer extends Layer {
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
  override def forward(downstream: Datum): Datum = ???


  /**
   * Calculate layer output with error differentiation
   * according to concrete implementation.
   * [[com.lewuathe.neurallib.layers.Layer.backward]] must be given
   * the datum which retain a [[com.lewuathe.neurallib.Channel]]
   * because of diff calculation.
   * @param upstream
   */
  override def backward(upstream: Datum): _root_.com.lewuathe.neurallib.Datum = ???

  /**
   * Calculate error update difference on each input error
   * for weights and biases
   * @param upstream
   * @return parameter delta
   */
  override def delta(upstream: Datum): Option[Datum] = ???

  override def isParamLayer: Boolean = true
}
