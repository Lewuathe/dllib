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

import breeze.linalg.{Vector => brzVector}

import com.lewuathe.dllib.{ActivationStack, Bias, Blob}
import com.lewuathe.dllib.{BlobShape, Weight}
import com.lewuathe.dllib.model.Model
import com.lewuathe.dllib.util.genId

/**
  * Convert multiple vector Blob into uni Blob.
  */
class FlattenLayer(override val outputShape: BlobShape,
                   override val inputShape: BlobShape)
    extends Layer
    with Visualizable {
  override var id: String = genId()

  override val inputSize: Int  = inputShape.featureSize
  override val outputSize: Int = outputShape.featureSize

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    *
    * @param acts
    * @param model
    * @return The output tuple of the layer.
    */
  override def forward(acts: ActivationStack, model: Model): Blob[Double] = {
    val input: Blob[Double] = acts.top
    require(input.size == inputShape.numChannel, "Invalid input shape")

    input.flatten
  }

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    *
    * @param delta
    * @param acts
    * @param model
    * @return The delta tuple of the layer while back propagation.
    *         First is passed previous layer, the second and third is
    *         the delta of Weight and Bias parameter of the layer.
    */
  override def backward(delta: Blob[Double],
                        acts: ActivationStack,
                        model: Model): (Blob[Double], Weight, Bias) = {
    val thisOutput = acts.pop()
    val thisInput  = acts.top

    require(delta.size == 1, "The delta is not flattened")

    // Convert uni blob into input shape blob
    val d = Blob[Double](
      delta.head.toArray
        .sliding(inputSize, inputSize)
        .map(a => brzVector(a))
        .toArray)

    // No necessary to train this layer.
    val dWeight = Weight.zero(id, outputSize, inputSize)
    val dBias   = Bias.zero(id, outputSize)

    (d, dWeight, dBias)
  }
}
