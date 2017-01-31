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

package com.lewuathe.dllib.model

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.Layer
import com.lewuathe.dllib.{Bias, Weight}

class ModelShape(graph: Graph) extends Serializable {
  val weightShape = graph.layers.map({
    case layer: Layer => (layer.id, layer.outputSize, layer.inputSize)
  })
  val biasShape = graph.layers.map({
    case layer: Layer => (layer.id, layer.outputSize)
  })
}

abstract class Model(val graph: Graph, isZero: Boolean)(
    implicit ws: Option[Map[String, Weight]],
    bs: Option[Map[String, Bias]]
) extends Serializable {
  def init(): (Map[String, Weight], Map[String, Bias])

  val shape: ModelShape = new ModelShape(graph)
  var (weights, biases) = if (ws.isEmpty && bs.isEmpty) {
    init()
  } else {
    (ws.get, bs.get)
  }

  def +(that: Model): Model

  def -(that: Model): Model

  def /(denom: Double): Model

  def *(times: Double): Model

  def +(that: Weight): Model

  def +(that: Bias): Model

  def getWeight(id: String): Option[Weight]

  def getBias(id: String): Option[Bias]

  def addWeight(w: Weight): Unit

  def addBias(b: Bias): Unit

  def contains(w: Weight): Boolean

  def contains(b: Bias): Boolean

  override def toString: String = {
    "Model\n  " +
      "  Weights\n" +
      weights.map({
        case (id, w) => s"    id=>${id}, weight=>${w}\n"
      }) +
      "  Biases\n" +
      biases.map({
        case (id, b) => s"    id=>${id}, bias=>${b}\n"
      })
  }
}
