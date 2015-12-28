package com.lewuathe.dllib.layer

import breeze.linalg.{Matrix, Vector}
import com.lewuathe.dllib.activations._
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util._

class ClassificationLayer(override val outputSize: Int,
                         override val inputSize: Int) extends Layer with Serializable {

  override val id: String = genId()

  override def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    require(weight.rows == outputSize, "Invalid weight output size")
    require(weight.cols == inputSize, "Invaid weight input size")
    require(bias.size == outputSize, "Invalid bias size")

    val (_, input) = acts.top
    require(input.size == inputSize, "Invalid input")
    val u: Vector[Double] = weight * input + bias
    val z = softmax(u)
    (u, z)
  }

  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], ActivationStack, Weight, Bias) = {
    require(delta.size == outputSize)
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val (thisU, thisZ) = acts.pop()
    val (backU, backZ) = acts.top

    val dWeight: Weight = new Weight(id, outputSize,
      inputSize)(delta.toDenseVector * backZ.toDenseVector.t)
    require(dWeight.value.rows == outputSize)
    require(dWeight.value.cols == inputSize)

    val dBias: Bias = new Bias(id, outputSize)(delta)
    require(dBias.value.size == outputSize)

    val d: Vector[Double] = sigmoidPrime(backU) :* (weight.toDenseMatrix.t * delta.toDenseVector)
    (d, acts, dWeight, dBias)
  }
}
