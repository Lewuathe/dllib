package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.activations.{sigmoid, sigmoidPrime}
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util.genId

class FullConnectedLayer(override val outputSize: Int,
                         override val inputSize: Int) extends Layer {

  override val id = genId()

  override def forward(acts: ActivationStack, model: Model): (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val (_, input) = acts.last
    val u = weight * input + bias
    val z = sigmoid(u)
    acts.push((u, z))
    (u, z)
  }

  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], ActivationStack, Weight, Bias) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val (thisU, thisZ) = acts.pop()
    val (backU, backZ) = acts.last
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
