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

import scala.util.control.Breaks._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import breeze.linalg.{Vector => brzVector}
import com.lewuathe.dllib.{ActivationStack, Instance}
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{Layer, PretrainLayer}
import com.lewuathe.dllib.model.{InMemoryModel, Model}
import com.lewuathe.dllib.util

/**
  * Pretrainer provides a way to train pre train networks before running
  * fully backpropagation. Pretrainer assumes pretrain layers are put
  * continuously at the head of network.
  */
trait Pretrainer extends Solver[Vector,
  UnsupervisedPretrainingSolver, UnsupervisedPretrainingSolverModel] {

  private def iteration(pretrainLayer: PretrainLayer, iter: Int,
                        instances: RDD[Instance], model: Model, graph: Graph,
                        pretrainTmpModel: Model, pretrainTmpGraph: Graph,
                        sc: SparkContext): (Model, Model) = {
    val bcModel = sc.broadcast(model)
    val bcPretrainTmpModel = sc.broadcast(pretrainTmpModel)
    val (modelDelta: Model, lossSum: Double, miniBatchSize: Int,
        pretrainTmpModelDelta: Model)
    = instances.sample(false, miniBatchFraction, 42 + iter)
      .treeAggregate(InMemoryModel.zero(graph), 0.0, 0, InMemoryModel.zero(pretrainTmpGraph))(
        seqOp = (c: (Model, Double, Int, Model), instance: Instance) => {
          // Sample feature
          val activations = new ActivationStack
          activations.push(instance.blob)

          // Feed forward to pretrained target layer
          breakable(
            for (l: Layer <- graph.layers) {
              // Target pretrain layer does not need to forward
              if (l.id == pretrainLayer.id) break
              val z = l.forward(activations, bcModel.value)
              activations.push(z)
            }
          )

          // g1 = (dWeight for hidden layer, dBias for hidden layer)
          // g2 = (dWeight for visible layer, dBias for visible layer)
          // g2 cannot be used unless the network is tied weight
          val (g1, g2, loss) = pretrainLayer.pretrain(activations,
            bcModel.value, bcPretrainTmpModel.value)
          (c._1 + g1._1 + g1._2, c._2 + loss, c._3 + 1, c._4 + g2._2)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3, c1._4 + c2._4)
        }
      )
    logInfo(s"Iteration ${iter} -> loss: ${lossSum / miniBatchSize}, " +
      s"count: ${miniBatchSize}, learning rate: ${learningRate}")
    (model + (modelDelta / miniBatchSize) * learningRate,
      pretrainTmpModel + (pretrainTmpModelDelta / miniBatchSize) * learningRate)
  }

  def pretrainInternal(dataset: Dataset[_], model: Model): Model = {
    val numFeatures = dataset.select(col($(featuresCol)))
      .first().getAs[Vector](0).size
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) {
      lit(1.0)
    } else {
      col($(weightCol))
    }

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) =>
        val l = util.encodeLabel(label, graph.layers.last.outputSize)
        Instance(l, weight, brzVector[Double](features.toArray))
    }

    var localModel = model
    val bcGraph = dataset.sqlContext.sparkContext.broadcast(graph)

    // TODO: Refactoring to be readable
    for (layer <- graph.layers if layer.isInstanceOf[PretrainLayer]) {
      // Pretraining can be applied only for PretrainLayer
      layer match {
        case pretrainLayer: PretrainLayer => {
          var (pretrainTmpModel, pretrainTmpForm)
            = createPretrainTmpNetwork(pretrainLayer)
          val bcPretrainTmpForm
            = dataset.sqlContext.sparkContext.broadcast(pretrainTmpForm)
          for (iter <- 0 until $(numIterations)) {
            val ret = iteration(pretrainLayer, iter, instances, localModel,
              bcGraph.value, pretrainTmpModel , bcPretrainTmpForm.value,
              instances.sparkContext)
            localModel = ret._1
            pretrainTmpModel = ret._2
            learningRate *= learningRateDecay
          }
        } // case
      } // match
    } // for
    localModel
  }

  // Tmp model represents a model that is only used while pretraining
  private def createPretrainTmpNetwork(pretrainLayer: PretrainLayer):
      (Model, Graph) = {
    val tmpForm = new Graph(Array(pretrainLayer.createTmpLayer()))
    (InMemoryModel(tmpForm), tmpForm)
  }
}
