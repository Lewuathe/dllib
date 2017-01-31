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

import breeze.linalg.{Matrix, Vector => brzVector}
import breeze.stats.distributions.Binomial

import com.lewuathe.dllib.{ActivationStack, Bias, Blob}
import com.lewuathe.dllib.{BlobShape, Weight}
import com.lewuathe.dllib.activations.sigmoid
import com.lewuathe.dllib.model.Model
import com.lewuathe.dllib.util.genId

class DenoisingAutoEncodeLayer(override val outputSize: Int,
                               override val inputSize: Int)
    extends PretrainLayer
    with ShapeValidator
    with Visualizable
    with UniBlobSupport {
  // Temporary ID used for storing pretrain parameters on Model
  override var id                     = genId()
  override val inputShape: BlobShape  = BlobShape(1, inputSize)
  override val outputShape: BlobShape = BlobShape(1, outputSize)

  val corruptionLevel = 0.7

  protected def corrupt(input: brzVector[Double]): brzVector[Double] = {
    val mask = brzVector(
      Binomial(1, 1.0 - corruptionLevel)
        .sample(input.length)
        .map(_.toDouble): _*)
    mask :* input
  }

  /**
    * Encode the input to hidden layer
    *
    * @param input
    * @param model
    * @param tmpModel
    * @return
    */
  override def encode(
      input: brzVector[Double],
      model: Model,
      tmpModel: Model): (brzVector[Double], brzVector[Double]) = {
    val weight: Matrix[Double]  = model.getWeight(id).get.value
    val bias: brzVector[Double] = model.getBias(id).get.value

    val u: brzVector[Double] = weight * corrupt(input) + bias
    val z                    = sigmoid(u)
    (u, z)
  }

  /**
    * Decode hidden layer value to visible layer
    *
    * @param input
    * @param model
    * @param tmpModel
    * @return
    */
  override def decode(
      input: brzVector[Double],
      model: Model,
      tmpModel: Model): (brzVector[Double], brzVector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    // Make sure to restore a Bias for pretrain visualization layer
    val bias: brzVector[Double] = tmpModel.getBias(id).get.value

    // TODO: decode bias should be stored in model
    val u: brzVector[Double] = weight.toDenseMatrix.t * input + bias
    val z                    = sigmoid(u)
    (u, z)
  }

  /**
    * Calculate the error of output layer between label data and prediction.
    *
    * @param label
    * @param prediction
    * @return
    */
  protected def error(label: brzVector[Double],
                      prediction: brzVector[Double]): brzVector[Double] = {
    require(label.size == prediction.size)
    val ret = label - prediction
    ret.map({
      case (d: Double) if d.isNaN => 0.0
      case (d: Double)            => d
    })
  }

  /**
    * Returns the form for creating tmp model used while pretraining
    * The layer used as prototype for creating tmp model. Only necessary
    * fields are input size, output size and id.
    *
    * @return A new pretrain layer that is reversed output and input.
    *         It is used mainly for keeping bias value while pretraining.
    */
  override def createTmpLayer: PretrainLayer = {
    val tmpLayer = new DenoisingAutoEncodeLayer(inputSize, outputSize)
    tmpLayer.id = this.id
    tmpLayer
  }

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return The output tuple of the layer. First value of the tuple
    *         represents the raw output, the second is applied activation
    *         function of the layer.
    */
  override def forward(acts: ActivationStack, model: Model): Blob[Double] = {
    val weight: Matrix[Double]  = model.getWeight(id).get.value
    val bias: brzVector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val input = acts.top
    checkBlobSize(input)
    require(input.head.size == inputSize, "Invalid input")

    val u: brzVector[Double] = weight * input.head + bias
    Blob.uni(u)
  }

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    *
    * @param delta
    * @param acts
    * @param model
    * @return
    */
  override def backward(delta: Blob[Double],
                        acts: ActivationStack,
                        model: Model): (Blob[Double], Weight, Bias) = {
    val weight: Matrix[Double]  = model.getWeight(id).get.value
    val bias: brzVector[Double] = model.getBias(id).get.value

    val thisOutput = acts.pop()
    val thisInput  = acts.top

    checkBlobSize(delta)

    val dWeight: Weight = new Weight(id, outputSize, inputSize)(
      Some(delta.head.toDenseVector * thisInput.head.toDenseVector.t))
    val dBias: Bias = new Bias(id, outputSize)(Some(delta.head))

    validateParamShapes(dWeight.value, dBias.value)

    val d: brzVector[Double] = weight.toDenseMatrix.t * delta.head
        .toDenseVector
    (Blob.uni(d), dWeight, dBias)
  }
}
