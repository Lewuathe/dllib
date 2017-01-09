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
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{AffineLayer, ReLULayer, SoftmaxLayer}
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.MultiLayerPerceptron
import com.lewuathe.dllib.Model

class MNISTApp(miniBatchFraction: Double, numIterations: Int, learningRate: Double) {
  def createMNISTDataset(path: String, sc: SparkContext): DataFrame = {
    val dataset = MNIST(path)
    MNIST.asDF(dataset, sc, 5000)
  }

  def submit(sc: SparkContext): Unit = {
    val sqlContext = new SQLContext(sc)
    val df = createMNISTDataset("/tmp/", sc)

    val nn3Graph = new Graph(Array(
      new AffineLayer(100, 784),
      new ReLULayer(100, 100),
      new AffineLayer(10, 100),
      new SoftmaxLayer(10, 10)
    ))

    val nn3Model = Model(nn3Graph)
    val nn3 = Network(nn3Model, nn3Graph)

    val multilayerPerceptron = new MultiLayerPerceptron("MNIST", nn3)
    multilayerPerceptron.miniBatchFraction = miniBatchFraction
    multilayerPerceptron.numIterations = numIterations
    multilayerPerceptron.learningRate = learningRate
    val model = multilayerPerceptron.fit(df)

    val result = model.transform(df)

    result.filter("label = prediction").count()
  }
}

object MNISTApp {
  def submit(sc: SparkContext): Unit = new MNISTApp(0.03, 10, 0.5).submit(sc)

  def apply(sc: SparkContext, miniBatchFraction: Double,
            numIterations: Int, learningRate: Double): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    new MNISTApp(miniBatchFraction, numIterations, learningRate).submit(sc)
  }
}
