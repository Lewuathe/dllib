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

import com.lewuathe.dllib.{ActivationStack, Bias, Blob}
import com.lewuathe.dllib.{BlobShape, Model, Weight}
import com.lewuathe.dllib.util.genId

/**
  * FullConnectedLayer is an intermediate layer used for updating all
  * parameters between every units.
  * @param outputSize
  * @param inputSize
  */
class AffineLayer(
      override val outputSize: Int,
      override val inputSize: Int)
    extends Layer with ShapeValidator with Visualizable with UniBlobSupport {

  override var id = genId()
  override val inputShape: BlobShape = BlobShape(1, inputSize)
  override val outputShape: BlobShape = BlobShape(1, outputSize)

  override def forward(acts: ActivationStack, model: Model): Blob[Double] = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val input: Blob[Double] = acts.top
    checkBlobSize(input)
    require(input.head.length == inputSize, "Invalid input")

    val u: Vector[Double] = weight * input.head + bias
    Blob.uni(u)
  }

  override def backward(delta: Blob[Double], acts: ActivationStack,
                        model: Model): (Blob[Double], Weight, Bias) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val thisOutput = acts.pop()
    val thisInput = acts.top

    checkBlobSize(delta)
    checkBlobSize(thisInput)

    val dWeight: Weight = new Weight(id, outputSize,
      inputSize)(delta.head.toDenseVector * thisInput.head.toDenseVector.t)
    val dBias: Bias = new Bias(id, outputSize)(delta.head)

    validateParamShapes(dWeight.value, dBias.value)

    val d: Vector[Double] = weight.toDenseMatrix.t * delta.head.toDenseVector
    (Blob.uni(d), dWeight, dBias)
  }
}
