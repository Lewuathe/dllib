package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.{Weight, Bias, ActivationStack, Model}
import com.lewuathe.dllib.activations._

abstract class PretrainLayer extends Layer with ShapeValidator {

  val pretrainId: String

  private def addPretrainParams(model: Model): Model = {
    // Add Pretrain bias that is not needed
    // any more when prediction phase
    model.addBias(Bias(pretrainId, inputSize))
    model
  }

  /**
    * Pretraining with input data to the layer.
    * We can assume AutoEncoder can inherit this class.
    * It returns the gradient of weight and bias of the layer.
    * @param acts
    * @param model
    * @return (dWeight1, dBias1): Hidden Layer Gradient
    *         (dWeight2, dBias2): Visible Layer Gradient
    *         (Hidden Layer Gradient, Visible Layer Gradient)
    */
  def pretrain(acts: ActivationStack, model: Model): ((Weight, Bias), (Weight, Bias), Double) = {
    val (_, input) = acts.top

    // Add a bias that is necessary to complete pretain but temporary
    if (model.contains(Bias.zero(pretrainId, inputSize))) addPretrainParams(model)

    val (hiddenU, hiddenZ) = encode(input, model)
    val (visibleU, visibleZ) = decode(hiddenZ, model)

    val delta2 = error(input, visibleZ)
    val loss = Math.sqrt((delta2 :* delta2).sum)

    // NOTE: Gradient of decode layer.
    // Make sure output and input is reversed
    val dWeight2: Weight = new Weight(id, inputSize, outputSize)(delta2.toDenseVector * hiddenZ.toDenseVector.t)
    val dBias2: Bias = new Bias(pretrainId, inputSize)(delta2)

    // Back propagation of delta
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val delta1: Vector[Double] = sigmoidPrime(hiddenU) :* (weight.toDenseMatrix * delta2.toDenseVector)

    val dWeight1: Weight = new Weight(id, outputSize, inputSize)(delta1.toDenseVector * input.toDenseVector.t)
    val dBias1: Bias = new Bias(id, outputSize)(delta1)
    validateParamShapes(dWeight1.value, dBias1.value)

    // (Hidden Layer Gradient, Visible Layer Gradient)
    ((dWeight1, dBias1), (dWeight2, dBias2), loss)
  }

  protected def error(input: Vector[Double], visible: Vector[Double]): Vector[Double]

  protected def encode(input: Vector[Double], model: Model): (Vector[Double], Vector[Double])
  protected def decode(input: Vector[Double], model: Model): (Vector[Double], Vector[Double])
}
