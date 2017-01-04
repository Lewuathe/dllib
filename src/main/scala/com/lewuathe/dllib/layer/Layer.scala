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

package com.lewuathe.dllib.layer

import breeze.linalg.Vector

import com.lewuathe.dllib.{ActivationStack, Bias, Model, Weight}

/**
  * Layer is an abstraction of neural network layer.
  * This class only retains the size of input and output
  * not coefficient and intercept. Actual parameters are kept in Model class.
  * The parameters can be accessed with id.
  */
abstract class Layer extends Serializable {
  var id: String
  val inputSize: Int
  val outputSize: Int

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return The output tuple of the layer.
    */
  def forward(acts: ActivationStack, model: Model)
      : Vector[Double]

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    * @param delta
    * @param acts
    * @param model
    * @return The delta tuple of the layer while back propagation.
    *         First is passed previous layer, the second and third is
    *         the delta of Weight and Bias parameter of the layer.
    */
  def backward(delta: Vector[Double], acts: ActivationStack, model: Model)
      : (Vector[Double], Weight, Bias)

  override def toString: String = {
    s"id: ${id}, ${inputSize} -> ${outputSize}"
  }
}
