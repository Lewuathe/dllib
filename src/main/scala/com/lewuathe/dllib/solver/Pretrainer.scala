package com.lewuathe.dllib.solver

import scala.util.control.Breaks._

import breeze.linalg.Vector
import com.lewuathe.dllib.layer.{Layer, PretrainLayer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.functions.{col, lit}

import com.lewuathe.dllib.{ActivationStack, Model, Instance, util}

trait Pretrainer extends Solver[org.apache.spark.mllib.linalg.Vector, UnsupervisedPretrainingSolver, UnsupervisedPretrainingSolverModel] {
  def pretrainInternal(dataset: DataFrame, model: Model): Model = {
    val w = lit(1.0)
    val instances: RDD[Instance] = dataset.select(col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: org.apache.spark.mllib.linalg.Vector) => {
        val l = util.encodeLabel(label, form.layers.last.outputSize)
        Instance(l, weight, Vector[Double](features.toArray))
      }
    }

    var localModel = model
    val bcForm = dataset.sqlContext.sparkContext.broadcast(form)

    // TODO: Refactoring to be readable
    for (layer <- form.layers) {
      layer match {
        case pretrainedLayer: PretrainLayer => {
          // Pretraining can be applied only for PretrainLayer
          for (iter <- 0 until numIterations) {
            val bcModel = dataset.sqlContext.sparkContext.broadcast(localModel)
            val (modelDelta: Model, lossSum: Double, miniBatchSize: Int)
            = instances.sample(false, miniBatchFraction, 42 + iter)
              .treeAggregate(Model.zero(form), 0.0, 0)(
                seqOp = (c: (Model, Double, Int), instance: Instance) => {
                  // Sample feature
                  val activations = new ActivationStack
                  activations.push((instance.features, instance.features))

                  // Feed forward to pretrained target layer
                  breakable(
                    for (l: Layer <- form.layers) {
                      if (l.id == pretrainedLayer.id) break
                      val (u, z) = l.forward(activations, bcModel.value)
                      activations.push((u, z))
                    }
                  )

                  // g1 = (dWeight for hidden layer, dBias for hidden layer)
                  // g2 = (dWeight for visible layer, dBias for visible layer)
                  // g2 cannot be used unless the network is tied weight
                  val (g1, g2, loss) = pretrainedLayer.pretrain(activations, bcModel.value)
                  (c._1 + g1._1 + g1._2, c._2 + loss, c._3 + 1)
                },
                combOp = (c1, c2) => {
                  (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
                }
              )
            logInfo(s"Iteration ${iter} -> loss: ${lossSum / miniBatchSize}, " +
              s"count: ${miniBatchSize}, learning rate: ${learningRate}")
            localModel += (modelDelta / miniBatchSize) * learningRate
            learningRate *= learningRateDecay
          }
        }
      }
    }
    localModel
  }
}
