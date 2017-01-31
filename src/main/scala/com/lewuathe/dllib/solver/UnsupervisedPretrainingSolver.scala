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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset

import breeze.linalg.{Vector => brzVector}

import com.lewuathe.dllib.Blob
import com.lewuathe.dllib.network.Network

class UnsupervisedPretrainingSolver(override val uid: String, network: Network)
    extends Solver[Vector,
                   UnsupervisedPretrainingSolver,
                   UnsupervisedPretrainingSolverModel](network)
    with Pretrainer {
  override def copy(extra: ParamMap): UnsupervisedPretrainingSolver =
    defaultCopy(extra)

  override protected def train(
      dataset: Dataset[_]): UnsupervisedPretrainingSolverModel = {
    logInfo(s"Pretraining...")
    val pretrainedModel = pretrainInternal(dataset, model)
    logInfo(s"Fine tuning")
    val newModel   = trainInternal(dataset, pretrainedModel)
    val newNetwork = new Network(newModel, network.graph)
    copyValues(new UnsupervisedPretrainingSolverModel(uid, newNetwork))
  }
}

class UnsupervisedPretrainingSolverModel(override val uid: String,
                                         network: Network)
    extends SolverModel[Vector, UnsupervisedPretrainingSolverModel](network) {
  override protected def predict(features: Vector): Double = {
    val brzFeatures = brzVector[Double](features.toArray)
    predictInternal(Blob.uni(brzFeatures))
  }
}
