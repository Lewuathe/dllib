package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.Datum

/**
 * Created by lewuathe on 5/13/15.
 */
abstract class Layer {
  private[neurallib] var param: Datum = _

//  private[neurallib] val paramStruct
//  = new Layer.ParamStruct(param.channels, param.height, param.width)

  private[neurallib] var prevForward: Datum = _

  def zeroParam: Datum = param.zero()

  /**
   * Initialize parameters of this layer
   */
  def init(conf: Map[String, String]): Unit

  /**
   * Calculate layer output according
   * to concrete implementation.
   * @param downstream
   * @return output Datum
   */
  def forward(downstream: Datum): Datum

  /**
   * Calculate layer output with error differentiation
   * according to concrete implementation.
   * [[com.lewuathe.neurallib.layers.Layer.backward]] must be given
   * the datum which retain a [[com.lewuathe.neurallib.DiffChannel]]
   * because of diff calculation.
   * @param upstream
   */
  def backward(upstream: Datum): Datum

  /**
   * Calculate error update difference on each input error
   * for weights and biases
   * @param upstream
   * @return parameter delta
   */
  def delta(upstream: Datum): Option[Datum]

  def isParamLayer: Boolean
}

object Layer {

  /**
   * This class represents the parameter structure of one layer.
   * @param channels
   * @param height
   * @param width
   */
  case class ParamStruct(channels: Int, height: Int, width: Int)
}