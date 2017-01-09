/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.lewuathe.dllib

import breeze.linalg.{sum, Vector}

/**
  * The collection of activation functions
  */
package object activations {
  /**
   * sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoid(x: Vector[Double]): Vector[Double] = x.map(sigmoid)

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
  def sigmoidPrime(x: Vector[Double]): Vector[Double] = x.map(sigmoidPrime)

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
  def tanh(x: Vector[Double]): Vector[Double] = x.map(tanh)

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
  def tanhPrime(x: Vector[Double]): Vector[Double] = x.map(tanhPrime)

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

  /**
    * Rectified linear unit activation function
    * @param input
    * @return
    */
  def relu(input: Double): Double = if (input > 0.0) input else 0.0

  def relu(x: Vector[Double]): Vector[Double] = x.map(relu)

  /**
    * Differential of Rectified linear unit activation function
    * @param input
    * @return
    */
  def reluPrime(input: Double): Double = if (input > 0.0) 1.0 else 0.0

  def reluPrime(x: Vector[Double]): Vector[Double] = x.map(reluPrime)
}
