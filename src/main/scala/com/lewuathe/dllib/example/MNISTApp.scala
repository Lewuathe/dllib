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

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{AffineLayer, ReLULayer, SoftmaxLayer}
import com.lewuathe.dllib.model.{InMemoryModel, Model}
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.MultiLayerPerceptron

class MNISTApp(miniBatchFraction: Double, numIter: Int, learningRate: Double) {
  var numSamples = 5000
  def createMNISTDataset(path: String, sc: SparkContext): DataFrame = {
    val dataset = MNIST(path)
    MNIST.asDF(dataset, sc, numSamples)
  }

  def submit(spark: SparkSession): Double = {
    val sqlContext = spark.sqlContext
    val df         = createMNISTDataset("/tmp/", spark.sparkContext)

    val nn3Graph = new Graph(
      Array(
        new AffineLayer(100, 784),
        new ReLULayer(100, 100),
        new AffineLayer(10, 100),
        new SoftmaxLayer(10, 10)
      ))

    val nn3Model = InMemoryModel(nn3Graph)
    val nn3      = Network(nn3Model, nn3Graph)

    val multilayerPerceptron = new MultiLayerPerceptron("MNIST", nn3)
    multilayerPerceptron.setNumIterations(numIter)
    multilayerPerceptron.miniBatchFraction = miniBatchFraction
    multilayerPerceptron.learningRate = learningRate
    val model = multilayerPerceptron.fit(df)

    val result = model.transform(df)

    result.filter("label = prediction").count() / numSamples.toDouble
  }
}

object MNISTApp {
  def submit(spark: SparkSession): Double =
    new MNISTApp(0.03, 10, 0.5).submit(spark)

  def apply(spark: SparkSession,
            miniBatchFraction: Double,
            numIterations: Int,
            learningRate: Double): Double = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    new MNISTApp(miniBatchFraction, numIterations, learningRate).submit(spark)
  }

  def apply(sparkConf: SparkConf,
            miniBatchFraction: Double,
            numIterations: Int,
            learningRate: Double): Double = {
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    new MNISTApp(miniBatchFraction, numIterations, learningRate).submit(spark)
  }
}
