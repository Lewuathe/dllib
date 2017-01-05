/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional ingraphation
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

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.Layer

class ModelShape(graph: Graph) extends Serializable {
  val weightShape = graph.layers.map({
    case layer: Layer => (layer.id, layer.outputSize, layer.inputSize)
  })
  val biasShape = graph.layers.map({
    case layer: Layer => (layer.id, layer.outputSize)
  })
}

class Model(graph: Graph, isZero: Boolean = false)
           (implicit ws: Map[String, Weight], bs: Map[String, Bias]) extends Serializable {
  val shape: ModelShape = new ModelShape(graph)

  def init(): (Map[String, Weight], Map[String, Bias]) = {
    val weights: Map[String, Weight] = graph.layers.map({
      case layer: Layer => {
        val w = Weight(layer.id, layer.outputSize, layer.inputSize, isZero)
        (w.id, w)
      }
    }).toMap

    val biases: Map[String, Bias] = graph.layers.map({
      case layer: Layer => {
        val b = Bias(layer.id, layer.outputSize, isZero)
        (b.id, b)
      }
    }).toMap
    (weights, biases)
  }

  var (weights, biases) = if (ws == null && bs == null) {
    init()
  } else {
    (ws, bs)
  }

  def +(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w + that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b + that.biases(id))
    })
    new Model(this.graph)(newWeights, newBiases)
  }

  def -(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w - that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b - that.biases(id))
    })
    new Model(this.graph)(newWeights, newBiases)
  }

  def /(denom: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w / denom)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b / denom)
    })
    new Model(this.graph)(newWeights, newBiases)
  }

  def *(times: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w * times)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b * times)
    })
    new Model(this.graph)(newWeights, newBiases)
  }

  def +(that: Weight): Model = {
    val oldWeight = this.weights.get(that.id).get
    this.weights += (that.id -> (oldWeight + that))
    this
  }

  def +(that: Bias): Model = {
    val oldBias = this.biases.get(that.id).get
    this.biases += (that.id -> (oldBias + that))
    this
  }

  def getWeight(id: String): Option[Weight] = weights.get(id)
  def getBias(id: String): Option[Bias] = biases.get(id)

  def addWeight(w: Weight): Unit = {
    require(!weights.contains(w.id))
    weights += (w.id -> w)
  }

  def addBias(b: Bias): Unit = {
    require(!biases.contains(b.id))
    biases += (b.id -> b)
  }

  def contains(w: Weight): Boolean = weights.contains(w.id)

  def contains(b: Bias): Boolean = biases.contains(b.id)


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

object Model {
  implicit val nullWeight: Map[String, Weight] = null
  implicit val nullBias : Map[String, Bias] = null

  def apply(graph: Graph): Model = new Model(graph)
  def zero(graph: Graph): Model = new Model(graph, isZero = true)
}
