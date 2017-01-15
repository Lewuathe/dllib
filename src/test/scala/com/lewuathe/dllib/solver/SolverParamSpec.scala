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

package com.lewuathe.dllib.solver

import com.lewuathe.dllib.network.Network
import org.scalamock.scalatest.MockFactory
import org.scalatest._

class SolverParamSpec extends FlatSpec with Matchers with MockFactory {
  val network = Network(null, null)

  "Solver" should "set label column name" in {
    val solver = new MultiLayerPerceptron("id", network)
    solver.setLabelCol("testLabel")
    solver.getLabelCol should be ("testLabel")
  }

  "Solver" should "set feature column name" in {
    val solver = new MultiLayerPerceptron("id", network)
    solver.setFeaturesCol("testFeature")
    solver.getFeaturesCol should be ("testFeature")
  }

  "Solver" should "set prediction column name" in {
    val solver = new MultiLayerPerceptron("id", network)
    solver.setPredictionCol("testPrediction")
    solver.getPredictionCol should be ("testPrediction")
  }

  "Solver" should "set number of iteration" in {
    val solver = new MultiLayerPerceptron("id", network)
    solver.setNumIterations(42)
    solver.getNumIterations should be (42)
  }
}
