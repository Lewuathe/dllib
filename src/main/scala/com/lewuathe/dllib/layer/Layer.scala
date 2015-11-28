package com.lewuathe.dllib.layer

import com.lewuathe.dllib.{ActivationStack, Model}
import org.apache.spark.mllib.linalg.Vector

abstract class Layer {
  val id: String
  val inputSize: Int
  val outputSize: Int

  def forward(acts: ActivationStack, model: Model): Vector
  def backward(delta: Vector, acts: ActivationStack, model: Model)
  : (Vector, ActivationStack)

}
