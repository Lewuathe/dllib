package com.lewuathe.neurallib

import com.lewuathe.neurallib.networks.Network
import org.apache.spark.rdd.RDD

/**
 * Created by lewuathe on 5/22/15.
 */
object Solver {
  private val numIterations: Int = 10
  private val miniBatchFraction: Double = 1.0

  def train(data: RDD[LabeledDatum], labels: Seq[Datum],
            net: Network): Seq[Datum] = {

    for (i <- 0 to numIterations) {
      val bcParams = data.context.broadcast(net.params)

      val (delta, lossSum, miniBatchSize)
        = data.sample(false, miniBatchFraction, 42 + i).
        treeAggregate((net.zeroParams, 0.0, 0))(
          seqOp = (c, d) => {
            // c: (delta, loss, count), d: (label, features)
            val thisDelta= net.delta(d)
            ((c._1 zip thisDelta).map(t => t._1 + t._2), 0.0, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c1, c2: (delta, loss, count)
            val d1 = c1._1
            val d2 = c2._1
            val count1 = c1._3
            val count2 = c2._3
            ((d1 zip d2).map(t => (t._1 / count1)
              + (t._2 / count2)), 0.0, count1 + count2)
          })

      // Update parameters of this network
      net.updateParams(delta)
    }

    // Return parameters of this model.
    net.params
  }

  def predict(data: RDD[Datum], net: Network): RDD[Datum]
    = data.map(net.forward)
}


