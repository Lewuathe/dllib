package com.lewuathe.neurallib

import breeze.linalg.{sum, DenseVector}

/**
 * Created by kaisasak on 6/29/14.
 */
package object activations {
  /**
   * sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoid(x: DenseVector[Double]): DenseVector[Double] = x.map(sigmoid(_))
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
  def sigmoidPrime(x: DenseVector[Double]): DenseVector[Double] = x.map(sigmoidPrime(_))

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
  def tanh(x: DenseVector[Double]): DenseVector[Double] = x.map(tanh(_))

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
  def tanhPrime(x: DenseVector[Double]): DenseVector[Double] = x.map(tanhPrime(_))

  /**
   * Derivative of hyperbolic tangent for double value
   * @param input
   * @return
   */
  def tanhPrime(input: Double): Double = 1.0 - Math.pow(tanh(input), 2.0)

  /**
   * Softmax activation function 
   * @param x
   */
  def softmax(x: DenseVector[Double]): DenseVector[Double] = {
    val denom = sum(x.map(Math.exp))
    x.map(Math.exp(_) / denom)
  }

  /**
   * A smooth approximation of ReLU
   * @param input
   * @return
   */
  def softplus(input: Double): Double = Math.log(1.0 + Math.exp(input))

  /**
   * A smooth approximation of ReLU for vector 
   * @param x
   * @return
   */
  def softplus(x: DenseVector[Double]): DenseVector[Double] = x.map(softplus(_))

  /**
   * Derivative of softplus function 
   * @param input
   * @return
   */
  def softplusPrime(input: Double): Double = 1.0 / (1.0 + Math.exp(-input))

  /**
   * Derivative of softplus function for vector
   * @param x
   * @return
   */
  def softplusPrime(x: DenseVector[Double]): DenseVector[Double] = x.map(softplusPrime(_))
  


}
