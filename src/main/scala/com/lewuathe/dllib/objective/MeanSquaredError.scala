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
package com.lewuathe.dllib.objective

import com.lewuathe.dllib.Blob
import com.lewuathe.dllib.layer.UniBlobSupport

class MeanSquaredError extends Objective with UniBlobSupport {


  /**
    * Calculate the difference between label vector and prediction vector.
    *
    * @param label      label vector
    * @param prediction prediction vector
    * @return the difference between two vectors
    */
  override def error(label: Blob[Double], prediction: Blob[Double]):
      Blob[Double] = {
    require(label.size == prediction.size)
    checkBlobSize(label)
    checkBlobSize(prediction)

    val ret = label.head - prediction.head
    Blob.uni(ret.map({
      case (d: Double) if d.isNaN => 0.0
      case (d: Double) => d
    }))
  }

  /**
    * Calculate the objective value which should be minimized with given
    * label vector and prediction vector.
    *
    * @param label      label vector
    * @param prediction prediction vector
    * @return the loss calculated with label and prediction
    */
  override def loss(label: Blob[Double], prediction: Blob[Double]): Double = {
    val delta = error(label, prediction)
    Math.sqrt((delta.head :* delta.head).sum)
  }
}

object MeanSquaredError {
  def apply(): MeanSquaredError = new MeanSquaredError
}
