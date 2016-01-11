package com.lewuathe.dllib.layer

import breeze.linalg.Vector

import com.lewuathe.dllib.{Weight, Bias, ActivationStack, Model}

/**
  * Layer is an abstraction of neural network layer.
  * This class only retains the size of input and output
  * not coefficient and intercept. Actual parameters are kept in Model class.
  * The parameters can be accessed with id.
  */
abstract class Layer extends Serializable {
  val id: String
  val inputSize: Int
  val outputSize: Int

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return
    */
  def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double])

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    * @param delta
    * @param acts
    * @param model
    * @return
    */
  def backward(delta: Vector[Double], acts: ActivationStack, model: Model)
  : (Vector[Double], Weight, Bias)

  override def toString: String = {
    s"id: ${id}, ${inputSize} -> ${outputSize}"
  }
}
