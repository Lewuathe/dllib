package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.activations.{sigmoid, sigmoidPrime}
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util.genId

class DenoisingAutoEncodeLayer(override val outputSize: Int,
                              override val inputSize: Int) extends PretrainLayer with ShapeValidator {
  override val id = genId()
  override val pretrainId = "p_" + id

  override def encode(input: Vector[Double], model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val u: Vector[Double] = weight * input + bias
    val z = sigmoid(u)
    (u, z)
  }

  override def decode(input: Vector[Double], model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    // Make sure to restore a Bias for pretrain visualization layer
    val bias: Vector[Double] = Bias(pretrainId, inputSize).value

    val u: Vector[Double] = weight.toDenseMatrix.t * input + bias
    val z = sigmoid(u)
    (u, z)
  }

  /**
    * Calculate the error of output layer between label data and prediction.
    * @param label
    * @param prediction
    * @return
    */
  protected def error(label: Vector[Double], prediction: Vector[Double]): Vector[Double] = {
    require(label.size == prediction.size)
    val ret = label - prediction
    ret.map({
      case (d: Double) if d.isNaN => 0.0
      case (d: Double) => d
    })
  }

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return
    */
  override def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val (_, input) = acts.top
    require(input.size == inputSize, "Invalid input")

    val u: Vector[Double] = weight * input + bias
    val z = sigmoid(u)
    (u, z)
  }

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    * @param delta
    * @param acts
    * @param model
    * @return
    */
  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], Weight, Bias) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val (thisU, thisZ) = acts.pop()
    val (backU, backZ) = acts.top

    val dWeight: Weight = new Weight(id, outputSize,
      inputSize)(delta.toDenseVector * backZ.toDenseVector.t)
    val dBias: Bias = new Bias(id, outputSize)(delta)

    validateParamShapes(dWeight.value, dBias.value)
    val d: Vector[Double] = sigmoidPrime(backU) :* (weight.toDenseMatrix.t * delta.toDenseVector)
    (d, dWeight, dBias)
  }
}
