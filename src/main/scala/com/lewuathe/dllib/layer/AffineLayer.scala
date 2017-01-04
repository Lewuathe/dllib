package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.activations.{sigmoid, sigmoidPrime}
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util.genId

/**
  * FullConnectedLayer is an intermediate layer used for updating all
  * parameters between every units.
  * @param outputSize
  * @param inputSize
  */
class AffineLayer(override val outputSize: Int,
                  override val inputSize: Int) extends Layer with ShapeValidator with Visualizable {

  override var id = genId()

  override def forward(acts: ActivationStack, model: Model): Vector[Double] = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val input = acts.top
    require(input.size == inputSize, "Invalid input")

    val u: Vector[Double] = weight * input + bias
    u
  }

  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], Weight, Bias) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val thisOutput = acts.pop()
    val thisInput = acts.top

    val dWeight: Weight = new Weight(id, outputSize,
      inputSize)(delta.toDenseVector * thisInput.toDenseVector.t)
    val dBias: Bias = new Bias(id, outputSize)(delta)

    validateParamShapes(dWeight.value, dBias.value)

    val d: Vector[Double] = weight.toDenseMatrix.t * delta.toDenseVector
    (d, dWeight, dBias)
  }

}
