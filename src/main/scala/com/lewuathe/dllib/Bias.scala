package com.lewuathe.dllib

import breeze.linalg.Vector

/**
  * Bias represents a intercept vector of each layer
  * @param id
  * @param size
  */
class Bias(val id: String, val size: Int, isZero: Boolean = false)
          (v: Vector[Double] = null) {

  val value: Vector[Double] = if (v != null) {
    v
  } else if (isZero) {
    zeroBias(size)
  } else {
    randomBias(size)
  }

  private def randomBias(size: Int): Vector[Double] = {
    Vector.rand[Double](size)
  }

  private def zeroBias(size: Int): Vector[Double] = {
    Vector.zeros[Double](size)
  }

  def +(that: Bias): Bias = {
    require(this.size == that.size)
    new Bias(id, size)(this.value + that.value)
  }
}

object Bias {
  def apply(id: String, size: Int): Bias = new Bias(id, size)

  def apply(size: Int): Bias = new Bias(util.genId(), size)

  def apply(id: String, size: Int, isZero: Boolean): Bias = new Bias(id, size, isZero)

  def zero(id: String, size: Int): Bias = new Bias(id, size, isZero = true)
}
