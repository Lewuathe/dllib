package com.lewuathe.dllib.layer

import breeze.linalg.Vector
import com.lewuathe.dllib.{ActivationStack, Bias, Model, Weight}
import com.lewuathe.dllib.util.genId
import com.lewuathe.dllib.activations.{softmax, sigmoidPrime}

/**
  * Sigmoid function layer
  */
class SoftmaxLayer(override val outputSize: Int,
                   override val inputSize: Int) extends Layer with Visualizable {
  override var id: String = genId()

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    *
    * @param acts
    * @param model
    * @return The output tuple of the layer. First value of the tuple
    *         represents the raw output, the second is applied activation
    *         function of the layer.
    */
  override def forward(acts: ActivationStack, model: Model): Vector[Double] = {
    val input = acts.top
    require(input.size == inputSize, "Invalid input")

    softmax(input)
  }

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    *
    * @param delta
    * @param acts
    * @param model
    * @return The delta tuple of the layer while back propagation.
    *         First is passed previous layer, the second and third is
    *         the delta of Weight and Bias parameter of the layer.
    */
  override def backward(delta: Vector[Double], acts: ActivationStack, model: Model): (Vector[Double], Weight, Bias) = {
    val thisOutput = acts.pop()
    val thisInput = acts.top

    // No necessary to train this layer.
    val dWeight = Weight.zero(id, outputSize, inputSize)
    val dBias = Bias.zero(id, outputSize)

    val d: Vector[Double] = sigmoidPrime(thisInput) :* delta.toDenseVector
    (d, dWeight, dBias)
  }
}
