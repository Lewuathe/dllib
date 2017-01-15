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

package com.lewuathe.dllib

import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.types.StructType

package object param {
  /**
    * Trait for shared param featuresCol (default: "features").
    */
  private[dllib] trait HasFeaturesCol extends Params {

    /**
      * Param for features column name.
      * @group param
      */
    final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "features column name")

    setDefault(featuresCol, "features")

    /** @group getParam */
    final def getFeaturesCol: String = $(featuresCol)
  }

  /**
    * Trait for shared param labelCol (default: "label").
    */
  private[dllib] trait HasLabelCol extends Params {

    /**
      * Param for label column name.
      * @group param
      */
    final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

    setDefault(labelCol, "label")

    /** @group getParam */
    final def getLabelCol: String = $(labelCol)
  }

  /**
    * Trait for shared param predictionCol (default: "prediction").
    */
  private[dllib] trait HasPredictionCol extends Params {

    /**
      * Param for prediction column name.
      * @group param
      */
    final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")

//    setDefault(predictionCol, "prediction")

    /** @group getParam */
    final def getPredictionCol: String = $(predictionCol)
  }

  /**
    * Trait for shared param weightCol.
    */
  private[dllib] trait HasWeightCol extends Params {

    /**
      * Param for weight column name. If this is not set or empty, we treat all instance weights as 1.0..
      * @group param
      */
    final val weightCol: Param[String] = new Param[String](this, "weightCol",
      "weight column name. If this is not set or empty, we treat all instance weights as 1.0.")

    /** @group getParam */
    final def getWeightCol: String = $(weightCol)
  }

  private[dllib] trait HasNumIterations extends Params {

    /**
      * Param for the number of iterations to be trained which should be positive integer.
      */
    final val numIterations: Param[Int] = new Param[Int](this, "numIterations",
      "Specify the count of iterations to be trained.")

    final def getNumIterations: Int = $(numIterations)
  }
}
