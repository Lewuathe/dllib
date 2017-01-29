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
import org.scalamock.scalatest.MockFactory
import org.scalatest._

import com.lewuathe.dllib.{ActivationStack, Blob, BlobShape}
import com.lewuathe.dllib.MockInMemoryModel
import com.lewuathe.dllib.graph.Graph


class FlattenLayerSpec extends FlatSpec with Matchers with MockFactory {
  val model = new MockInMemoryModel(new Graph(Array()))
  val inputShape = BlobShape(3, 3)
  val outputShape = BlobShape(1, 9)

  val layer = new FlattenLayer(outputShape, inputShape)

  "FlattenLayer" should "flatten multiple Blob input" in {
    val input = Blob[Double](Array(
      brzVector(1.0, 2.0, 3.0),
      brzVector(2.0, 3.0, 4.0),
      brzVector(3.0, 4.0, 5.0)
    ))

    val a = new ActivationStack
    a.push(input)
    val output = layer.forward(a, model)
    output.size should be(1)
    output.head should be(brzVector(1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0))
  }

  "FlattenLayer" should "backpropagate delta" in {
    val input = Blob[Double](Array(
      brzVector(1.0, 2.0, 3.0),
      brzVector(2.0, 3.0, 4.0),
      brzVector(3.0, 4.0, 5.0)
    ))
    val delta = Blob.uni(brzVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))

    val a = new ActivationStack
    a.push(input).push(input)
    val (d, _, _) = layer.backward(delta, a, model)
    d.size should be(3)
    d.channel(0) should be(brzVector(1.0, 2.0, 3.0))
    d.channel(1) should be(brzVector(4.0, 5.0, 6.0))
    d.channel(2) should be(brzVector(7.0, 8.0, 9.0))
  }
}
