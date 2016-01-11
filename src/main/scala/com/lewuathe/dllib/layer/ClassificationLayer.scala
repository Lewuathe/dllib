package com.lewuathe.dllib.layer

import breeze.linalg.{Matrix, Vector}
import com.lewuathe.dllib.activations._
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util._

/**
  * Simple output layer for multi-class classification problem.
  * It uses softmax function as activation function of output.
  * @param outputSize
  * @param inputSize
  */
class ClassificationLayer(override val outputSize: Int,
                         override val inputSize: Int) extends Layer with ShapeValidator {

  override val id: String = genId()

  override def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val (_, input) = acts.top
    require(input.size == inputSize, "Invalid input")
    val u: Vector[Double] = weight * input + bias
    val z = softmax(u)
    (u, z)
  }

  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], Weight, Bias) = {
    require(delta.size == outputSize)
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
