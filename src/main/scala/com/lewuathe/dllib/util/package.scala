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

import breeze.linalg.{DenseVector, Vector}

package object util {
  private val ID_SIZE = 16

  private val random = new scala.util.Random

  // Generate a random string of length n from the given alphabet
  private def randomString(alphabet: String)(n: Int): String =
    Stream.continually(random.nextInt(alphabet.size))
      .map(alphabet).take(n).mkString

  // Generate a random alphabnumeric string of length n
  private def randomAlphanumericString(n: Int) =
    randomString("abcdefghijklmnopqrstuvwxyz0123456789")(n)

  def genId(): String = randomAlphanumericString(ID_SIZE)

  def encodeLabel(label: Double, labelCount: Int): Vector[Double] = {
    val output = Array.fill(labelCount)(0.0)
    output(label.toInt) = 1.0
    Vector(output)
  }

  /**
    * Converts a vector to a label.
    * Returns the position of the maximal element of a vector.
    *
    * @param output label encoded with a vector
    * @return label
    */
  def decodeLabel(output: Blob[Double]): Double = {
    require(output.size == 1)
    output.channel(0).argmax.toDouble
  }

  def oneHotLabel(output: Vector[Double]): Vector[Double] = {
    val i = output.argmax
    val ret = DenseVector.zeros[Double](output.size)
    ret(i) = 1.0
    ret
  }
}
