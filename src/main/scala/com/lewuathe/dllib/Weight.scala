package com.lewuathe.dllib

import breeze.linalg.Matrix

/**
  * Weight represents a coefficient of each layer
  * @param id
  * @param inputSize
  * @param outputSize
  */
class Weight(val id: String, val outputSize: Int, val inputSize: Int, isZero: Boolean = false)
            (v: Matrix[Double] = null) {

  val value: Matrix[Double] = if (v != null) {
    v
  } else if (isZero) {
    zeroWeight(outputSize, inputSize)
  } else {
    randomWeight(outputSize, inputSize)
  }

  private def randomWeight(outputSize: Int, inputSize: Int): Matrix[Double] = {
    Matrix.rand[Double](outputSize, inputSize) - 0.5
  }

  private def zeroWeight(outputSize: Int, inputSize: Int): Matrix[Double] = {
    Matrix.zeros(outputSize, inputSize)
  }

  def +(that: Weight): Weight = {
    require(this.outputSize == that.outputSize)
    require(this.inputSize == that.inputSize)
    new Weight(id, outputSize, inputSize)(this.value + that.value)
  }
}

object Weight {
  def apply(id: String, outputSize: Int, inputSize: Int): Weight
    = new Weight(id, outputSize, inputSize)

  def apply(outputSize: Int, inputSize: Int): Weight
    = new Weight(util.genId(), outputSize, inputSize)

  def apply(id: String, outputSize: Int, inputSize: Int, isZero: Boolean): Weight
    = new Weight(id, outputSize, inputSize, isZero)

  def zero(id: String, outputSize: Int, inputSize: Int): Weight
    = new Weight(id, outputSize, inputSize, isZero = true)
}
