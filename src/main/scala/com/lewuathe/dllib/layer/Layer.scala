package com.lewuathe.dllib.layer

import com.lewuathe.dllib.{Weight, Bias, ActivationStack, Model}
import breeze.linalg.Vector

abstract class Layer {
  val id: String
  val inputSize: Int
  val outputSize: Int

  def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double])
  def backward(delta: Vector[Double], acts: ActivationStack, model: Model)
  : (Vector[Double], ActivationStack, Weight, Bias)

}
