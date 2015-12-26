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
    final val weightCol: Param[String] = new Param[String](this, "weightCol", "weight column name. If this is not set or empty, we treat all instance weights as 1.0.")

    /** @group getParam */
    final def getWeightCol: String = $(weightCol)
  }
}
