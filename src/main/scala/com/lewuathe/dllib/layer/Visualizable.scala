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

import breeze.linalg.{DenseMatrix, Matrix}
import breeze.plot._

import com.lewuathe.dllib.model.{InMemoryModel, Model}

trait Visualizable extends Layer {
  def vizWeight(outputPath: String, model: Model): Unit = {
    val weight = model.getWeight(id).get.value
    convertToImage(outputPath, weight)
  }

  private def convertToImage(
      outputPath: String,
      weight: Matrix[Double]): Unit = {
    val fWeight = Figure()

    val denseWeight: DenseMatrix[Double] = weight.toDenseMatrix
    for (i <- 0 until denseWeight.rows) {
      val (outputHeight, outputWidth) = calculateWindow(outputSize)
      val (inputHeight, inputWidth) = calculateWindow(inputSize)
      val reshaped = denseWeight(i, ::).t.toDenseMatrix
        .reshape(inputHeight, inputWidth)
      fWeight.subplot(outputHeight, outputWidth, i) += image(reshaped)
    }
    fWeight.saveas(outputPath)
  }

  /**
    * Calculate visualizable window size
    *
    * @return
    */
  private def calculateWindow(size: Int): (Int, Int) = {
    val height = Math.sqrt(size)
    val width = size / height
    (height.toInt, width.toInt)
  }
}
