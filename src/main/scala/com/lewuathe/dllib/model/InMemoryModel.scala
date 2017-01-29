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

class InMemoryModel(
    graph: Graph,
    isZero: Boolean = false)(
  implicit ws: Option[Map[String, Weight]], bs: Option[Map[String, Bias]]
) extends Model(graph, isZero)(ws, bs) {

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

  override def +(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w + that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b + that.biases(id))
    })
    new InMemoryModel(this.graph)(Some(newWeights), Some(newBiases))
  }

  override def -(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w - that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b - that.biases(id))
    })
    new InMemoryModel(this.graph)(Some(newWeights), Some(newBiases))
  }

  override def /(denom: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w / denom)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b / denom)
    })
    new InMemoryModel(this.graph)(Some(newWeights), Some(newBiases))
  }

  override def *(times: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w * times)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b * times)
    })
    new InMemoryModel(this.graph)(Some(newWeights), Some(newBiases))
  }

  override def +(that: Weight): Model = {
    val oldWeight = this.weights.get(that.id).get
    this.weights += (that.id -> (oldWeight + that))
    this
  }

  override def +(that: Bias): Model = {
    val oldBias = this.biases.get(that.id).get
    this.biases += (that.id -> (oldBias + that))
    this
  }

  override def getWeight(id: String): Option[Weight] = weights.get(id)
  override def getBias(id: String): Option[Bias] = biases.get(id)

  override def addWeight(w: Weight): Unit = {
    require(!weights.contains(w.id))
    weights += (w.id -> w)
  }

  override def addBias(b: Bias): Unit = {
    require(!biases.contains(b.id))
    biases += (b.id -> b)
  }

  override def contains(w: Weight): Boolean = weights.contains(w.id)

  override def contains(b: Bias): Boolean = biases.contains(b.id)

}

object InMemoryModel {
  implicit val nullWeight: Option[Map[String, Weight]] = Option.empty
  implicit val nullBias : Option[Map[String, Bias]] = Option.empty

  def apply(graph: Graph): Model = new InMemoryModel(graph)
  def zero(graph: Graph): Model = new InMemoryModel(graph, isZero = true)
}
