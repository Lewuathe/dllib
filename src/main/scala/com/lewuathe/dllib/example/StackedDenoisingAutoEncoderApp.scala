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

import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.Model
import com.lewuathe.dllib.layer.{AffineLayer, DenoisingAutoEncodeLayer, SigmoidLayer, SoftmaxLayer}
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.UnsupervisedPretrainingSolver

class StackedDenoisingAutoEncoderApp(miniBatchFraction: Double,
                                     numIterations: Int, learningRate: Double) {
  def createMNISTDataset(path: String, sc: SparkContext): DataFrame = {
    val dataset = MNIST(path)
    MNIST.asDF(dataset, sc, 5000)
  }

  def submit(sc: SparkContext): Unit = {
    val sqlContext = new SQLContext(sc)
    val df = createMNISTDataset("/tmp/", sc)

    val sdaForm = new Graph(Array(
      new DenoisingAutoEncodeLayer(100, 784),
      new SigmoidLayer(100, 100),
      new AffineLayer(10, 100),
      new SoftmaxLayer(10, 10)
    ))

    val sdaModel = Model(sdaForm)
    val sda = Network(sdaModel, sdaForm)

    val unsupervisedPretrainer = new UnsupervisedPretrainingSolver("MNIST", sda)
    unsupervisedPretrainer.miniBatchFraction = miniBatchFraction
    unsupervisedPretrainer.numIterations = numIterations
    unsupervisedPretrainer.learningRate = learningRate
    val model = unsupervisedPretrainer.fit(df)

    sdaForm.layers.foreach({
      case l: DenoisingAutoEncodeLayer => l.vizWeight("./images/weight_denoising.png", model.model)
    })

    val result = model.transform(df)

    result.filter("label = prediction").count()
  }
}

object StackedDenoisingAutoEncoderApp {
  def submit(sc: SparkContext): Unit
    = new StackedDenoisingAutoEncoderApp(0.03, 10, 0.5).submit(sc)

  def apply(sc: SparkContext, miniBatchFraction: Double,
            numIterations: Int, learningRate: Double): Unit = {
    new StackedDenoisingAutoEncoderApp(miniBatchFraction, numIterations, learningRate).submit(sc)
  }
}


