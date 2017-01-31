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

package com.lewuathe.dllib.example

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{AffineLayer, SoftmaxLayer}
import com.lewuathe.dllib.model.InMemoryModel
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.MultiLayerPerceptron

object XORApp {
  case class Sample(label: Double, features: Vector)
  def createTrainingData(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    val sc = sqlContext.sparkContext
    val data = sc
      .parallelize(
        Seq(
          (1.0, Array(1.0, 0.0)),
          (1.0, Array(0.0, 1.0)),
          (0.0, Array(0.0, 0.0)),
          (0.0, Array(1.0, 1.0))
        ))
      .map({
        case (label, features) => Sample(label, Vectors.dense(features))
      })
    data.toDF()
  }

  var miniBatchFraction = 1.0
  var numIterations     = 100
  var learningRate      = 0.7

  def submit(sc: SparkContext): Unit = {
    val sqlContext = new SQLContext(sc)

    val df = createTrainingData(sqlContext)

    val nn3Graph = new Graph(
      Array(
        new AffineLayer(2, 2),
        new SoftmaxLayer(2, 2)
      ))

    val nn3Model = InMemoryModel(nn3Graph)
    val nn3      = Network(nn3Model, nn3Graph)

    val multilayerPerceptron = new MultiLayerPerceptron("XOR", nn3)
    multilayerPerceptron.setNumIterations(numIterations)
    multilayerPerceptron.miniBatchFraction = miniBatchFraction
    multilayerPerceptron.learningRate = learningRate
    val model = multilayerPerceptron.fit(createTrainingData(sqlContext))

    val testData = Seq(
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0),
      Array(1.0, 1.0)
    )
    val result = model.transform(createTrainingData(sqlContext))

    result.show()
  }

  def main(args: Array[String]) {}
}
