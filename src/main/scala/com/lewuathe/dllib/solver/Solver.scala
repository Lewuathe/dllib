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

package com.lewuathe.dllib.solver

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import breeze.linalg.{Vector => brzVector}
import com.lewuathe.dllib.{ActivationStack, Instance, Model}
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.Layer
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.objective.{MeanSquaredError, Objective}
import com.lewuathe.dllib.param.HasWeightCol
import com.lewuathe.dllib.util

/**
  * Solver implements distributed training algorithm for deep learning models.
  * Currently this class is doing Back propagation under data parallelism schema.
  * @param network
  * @tparam FeaturesType
  * @tparam E
  * @tparam M
  */
abstract class Solver[FeaturesType,
                      E <: Solver[FeaturesType, E, M],
                      M <: SolverModel[FeaturesType, M]](val network: Network)
  extends Predictor[FeaturesType, E, M] with HasWeightCol {

  val graph: Graph = network.graph
  val model: Model = network.model

  logInfo(network.toString)

  var miniBatchFraction = 1.0
  var numIterations = 10
  var learningRate = 0.3
  val objective: Objective = new MeanSquaredError

  val learningRateDecay = 0.99

  protected def trainInternal(dataset: Dataset[_], model: Model): Model = {
    val numFeatures = dataset.select(col($(featuresCol))).first().getAs[Vector](0).size
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) =>
        val l = util.encodeLabel(label, graph.layers.last.outputSize)
        Instance(l, weight, brzVector[Double](features.toArray))
    }

    var localModel = model
    val bcGraph = dataset.sqlContext.sparkContext.broadcast(graph)

    for (i <- 0 until numIterations) {
      val bcModel = dataset.sqlContext.sparkContext.broadcast(localModel)
      val (modelDelta: Model, lossSum: Double, miniBatchSize: Int)
      = instances.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((Model.zero(graph), 0.0, 0))(
          seqOp = (c: (Model, Double, Int), instance: Instance) => {
            val (dModel, loss) = gradient(bcGraph.value, bcModel.value, instance)
            (c._1 + dModel, c._2 + loss, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // (Model, loss, count)
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      logInfo(s"Iteration ${i} -> loss: ${lossSum / miniBatchSize}, " +
        s"count: ${miniBatchSize}, learning rate: ${learningRate}")
      localModel += (modelDelta / miniBatchSize) * learningRate
    }

    localModel
  }

  /**
    * Calculate the gradient of Model parameter with given training instance.
    * @param form
    * @param model
    * @param instance
    * @return
    */
  protected def gradient(form: Graph, model: Model, instance: Instance): (Model, Double) = {
    var deltaModel = Model.zero(form)
    val label = instance.label
    val activations = new ActivationStack
    activations.push(instance.features)

    // Feed forward
    for (l: Layer <- form.layers) {
      val z = l.forward(activations, model)
      activations.push(z)
    }

    var delta = objective.error(label, activations.top)
    val loss = objective.loss(label, activations.top)

    // Back propagation
    for (l: Layer <- form.layers.reverse) {
      val (d, dWeight, dBias) = l.backward(delta, activations, model)
      delta = d
      deltaModel += dWeight
      deltaModel += dBias
    }

    (deltaModel, loss)
  }
}

abstract class SolverModel[FeaturesType, M <: SolverModel[FeaturesType, M]](val network: Network)
  extends PredictionModel[FeaturesType, M] {

  val model: Model = network.model
  val graph: Graph = network.graph

  protected def predictInternal(features: brzVector[Double]): Double = {
    val activations = new ActivationStack
    activations.push(features)
    // Feed forward
    for (l: Layer <- graph.layers) {
      val z = l.forward(activations, model)
      activations.push(z)
    }
    val ret = activations.top
    util.decodeLabel(ret)
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra)
}
