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

import breeze.linalg.Matrix

/**
  * Weight represents a coefficient of each layer
  * @param id
  * @param inputSize
  * @param outputSize
  */
class Weight(val id: String,
             val outputSize: Int,
             val inputSize: Int,
             isZero: Boolean = false)(implicit v: Option[Matrix[Double]])
    extends Serializable {

  val value: Matrix[Double] = if (v.isDefined) {
    v.get
  } else if (isZero) {
    zeroWeight(outputSize, inputSize)
  } else {
    randomWeight(outputSize, inputSize)
  }

  private def randomWeight(outputSize: Int, inputSize: Int): Matrix[Double] = {
    Matrix.rand[Double](outputSize, inputSize) - 0.5
  }

  private def zeroWeight(outputSize: Int, inputSize: Int): Matrix[Double] = {
    Matrix.zeros(outputSize, inputSize)
  }

  def +(that: Weight): Weight = {
    require(this.outputSize == that.outputSize)
    require(this.inputSize == that.inputSize)
    new Weight(id, outputSize, inputSize)(Some(this.value + that.value))
  }

  def -(that: Weight): Weight = {
    require(this.outputSize == that.outputSize)
    require(this.inputSize == that.inputSize)
    new Weight(id, outputSize, inputSize)(Some(this.value - that.value))
  }

  def /(denom: Double): Weight = {
    new Weight(id, outputSize, inputSize)(Some(this.value / denom))
  }

  def *(times: Double): Weight = {
    new Weight(id, outputSize, inputSize)(Some(this.value * times))
  }

  override def toString: String = {
    s"Weight: ${id} -> ${value}"
  }
}

object Weight {
  implicit val nullMatrix: Option[Matrix[Double]] = Option.empty

  def apply(id: String, outputSize: Int, inputSize: Int): Weight =
    new Weight(id, outputSize, inputSize)

  def apply(outputSize: Int, inputSize: Int): Weight =
    new Weight(util.genId(), outputSize, inputSize)

  def apply(id: String,
            outputSize: Int,
            inputSize: Int,
            isZero: Boolean): Weight =
    new Weight(id, outputSize, inputSize, isZero)

  def zero(id: String, outputSize: Int, inputSize: Int): Weight =
    new Weight(id, outputSize, inputSize, isZero = true)
}
