package com.lewuathe.dllib

import breeze.linalg.{sum, Vector}

/**
 * Created by kaisasak on 6/29/14.
 */
package object activations {
  /**
   * sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoid(x: Vector[Double]): Vector[Double] = x.map(sigmoid(_))

  /**
   * sigmoid function for double value
   * @param input
   * @return
   */
  def sigmoid(input: Double): Double = 1.0 / (1.0 + Math.exp(-input))

  /**
   * Derivative of sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoidPrime(x: Vector[Double]): Vector[Double] = x.map(sigmoidPrime(_))

  /**
   * Derivative of sigmoid function for double value
   * @param input
   * @return
   */
  def sigmoidPrime(input: Double): Double = sigmoid(input) * (1.0 - sigmoid(input))

  /**
   * Hyperbolic tangent for vector
   * @param x
   * @return
   */
  def tanh(x: Vector[Double]): Vector[Double] = x.map(tanh(_))

  /**
   * Hyperbolic tangent for double value
   * @param input
   * @return
   */
  def tanh(input: Double): Double = Math.tanh(input)

  /**
   * Derivative of hyperbolic tangent for vector
   * @param x
   * @return
   */
  def tanhPrime(x: Vector[Double]): Vector[Double] = x.map(tanhPrime(_))

  /**
   * Derivative of hyperbolic tangent for double value
   * @param input
   * @return
   */
  def tanhPrime(input: Double): Double = 1.0 - Math.pow(tanh(input), 2.0)

  def softmax(x: Vector[Double]): Vector[Double] = {
    val denom = sum(x.map(Math.exp))
    x.map(Math.exp(_) / denom)
  }
}
