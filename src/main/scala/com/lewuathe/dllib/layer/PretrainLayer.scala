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

import breeze.linalg.{Matrix, Vector}
import com.lewuathe.dllib.{ActivationStack, Bias, Weight}
import com.lewuathe.dllib.activations.sigmoidPrime
import com.lewuathe.dllib.model.Model

abstract class PretrainLayer extends Layer
  with ShapeValidator with UniBlobSupport {

  /**
    * Pretraining with input data to the layer.
    * We can assume AutoEncoder can inherit this class.
    * It returns the gradient of weight and bias of the layer.
 *
    * @param acts
    * @param model
    * @return (dWeight1, dBias1): Hidden Layer Gradient
    *         (dWeight2, dBias2): Visible Layer Gradient
    *         (Hidden Layer Gradient, Visible Layer Gradient)
    */
  def pretrain(acts: ActivationStack, model: Model, tmpModel: Model):
      ((Weight, Bias), (Weight, Bias), Double) = {
    val input = acts.top

    checkBlobSize(input)

    val (hiddenU, hiddenZ) = encode(input.head, model, tmpModel)
    val (visibleU, visibleZ) = decode(hiddenZ, model, tmpModel)

    val delta2 = error(input.head, visibleZ)
    val loss = Math.sqrt((delta2 :* delta2).sum)

    // NOTE: Gradient of decode layer.
    // Make sure output and input is reversed
    val dWeight2: Weight = new Weight(id, inputSize,
      outputSize)(delta2.toDenseVector * hiddenZ.toDenseVector.t)
    val dBias2: Bias = new Bias(id, inputSize)(delta2)

    // Back propagation of delta
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val delta1: Vector[Double] = sigmoidPrime(hiddenU) :*
      (weight.toDenseMatrix * delta2.toDenseVector)

    val dWeight1: Weight = new Weight(id, outputSize,
      inputSize)(delta1.toDenseVector * input.head.toDenseVector.t)
    val dBias1: Bias = new Bias(id, outputSize)(delta1)
    validateParamShapes(dWeight1.value, dBias1.value)

    // (Hidden Layer Gradient, Visible Layer Gradient)
    ((dWeight1, dBias1), (dWeight2, dBias2), loss)
  }

  /**
    * Returns the form for creating tmp model used while pretraining
    * @return
    */
  def createTmpLayer(): PretrainLayer

  protected def error(input: Vector[Double], visible: Vector[Double]):
    Vector[Double]

  /**
    * Encode the input toward hidden layer
    * @param input
    * @param model
    * @param tmpModel
    * @return The value of hidden layer
    */
  protected def encode(input: Vector[Double], model: Model, tmpModel: Model):
    (Vector[Double], Vector[Double])

  /**
    * Decode the value in hidden layer toward visible layer
    * @param input
    * @param model
    * @param tmpModel
    * @return The value of visible layer
    */
  protected def decode(input: Vector[Double], model: Model, tmpModel: Model):
    (Vector[Double], Vector[Double])
}
