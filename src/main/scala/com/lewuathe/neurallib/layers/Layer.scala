package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.Datum

/**
 * Created by kaisasak on 5/13/15.
 */
abstract class Layer {
  private[neurallib] var param: Datum = _

  private[neurallib] val paramStruct
  = new Layer.ParamStruct(param.channels, param.height, param.width)

  def zeroParam: Datum = param.zero()

  /**
   * Calculate layer output according
   * to concrete implementation.
   * @param input
   * @return output Datum
   */
  def forward(input: Datum): Datum

  /**
   * Calculate layer output with error differentiation
   * according to concrete implementation.
   * @param input
   */
  def backward(input: Datum): Datum

  /**
   * Calculate error update difference on each input error
   * for weights and biases
   * @param input
   * @return parameter delta
   */
  def delta(input: Datum): Datum
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