package com.lewuathe.dllib

import scala.util.Random

import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
  * Bias represents a intercept vector of each layer
  * @param id
  * @param size
  */
class Bias(val id: String, val size: Int, isZero = false: Boolean) {

  val value: Vector = if (isZero) {
    zeroBias(size)
  } else {
    randomBias(size)
  }

  private def randomBias(size: Int): Vector = {
    Vectors.dense(Array.fill(size)(Random.nextDouble()))
  }

  private def zeroBias(size: Int): Vector = {
    Vectors.zeros(size)
  }
}

object Bias {
  def apply(id: String, size: Int): Bias = new Bias(id, size)

  def apply(size: Int): Bias = new Bias(util.genId(), size)

  def apply(id: String, size: Int, isZero: Boolean): Bias = new Bias(id, size, isZero)

  def zero(id: String, size: Int): Bias = new Bias(id, size, isZero = true)
}
