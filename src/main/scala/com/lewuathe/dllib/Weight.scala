package com.lewuathe.dllib

import java.util.Random

import org.apache.spark.mllib.linalg.{Matrices, Matrix}

/**
  * Weight represents a coefficient of each layer
  * @param id
  * @param inputSize
  * @param outputSize
  */
class Weight(val id: String, val outputSize: Int, val inputSize: Int, isZero = false: Boolean) {

  val value: Matrix = if (isZero) {
    zeroWeight(outputSize, inputSize)
  } else {
    randomWeight(outputSize, inputSize)
  }

  private def randomWeight(outputSize: Int, inputSize: Int): Matrix = {
    Matrices.rand(outputSize, inputSize, new Random(42)) - 0.5
  }

  private def zeroWeight(outputSize: Int, inputSize: Int): Matrix = {
    Matrices.zeros(outputSize, inputSize)
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
