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

import breeze.linalg.Vector

/**
  * Bias represents a intercept vector of each layer
  * @param id
  * @param size
  */
class Bias(val id: String, val size: Int, isZero: Boolean = false)
          (implicit v: Vector[Double]) extends Serializable {

  val value: Vector[Double] = if (v != null) {
    v
  } else if (isZero) {
    zeroBias(size)
  } else {
    randomBias(size)
  }

  private def randomBias(size: Int): Vector[Double] = {
    Vector.rand[Double](size) - 0.5
  }

  private def zeroBias(size: Int): Vector[Double] = {
    Vector.zeros[Double](size)
  }

  def +(that: Bias): Bias = {
    require(this.size == that.size)
    new Bias(id, size)(this.value + that.value)
  }

  def -(that: Bias): Bias = {
    require(this.size == that.size)
    new Bias(id, size)(this.value - that.value)
  }

  def /(denom: Double): Bias = {
    new Bias(id, size)(this.value / denom)
  }

  def *(times: Double): Bias = {
    new Bias(id, size)(this.value * times)
  }

  override def toString: String = {
    s"Bias ${id} -> ${value}"
  }
}

object Bias {
  implicit val nullVector: Vector[Double] = null

  def apply(id: String, size: Int): Bias = new Bias(id, size)

  def apply(size: Int): Bias = new Bias(util.genId(), size)

  def apply(id: String, size: Int, isZero: Boolean): Bias = new Bias(id, size, isZero)

  def zero(id: String, size: Int): Bias = new Bias(id, size, isZero = true)
}
