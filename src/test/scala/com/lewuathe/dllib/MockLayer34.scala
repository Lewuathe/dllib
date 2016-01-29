package com.lewuathe.dllib

import breeze.linalg

import com.lewuathe.dllib.util._
import com.lewuathe.dllib.layer.Layer

class MockLayer34 extends Layer {
  override var id: String = "layer34"

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return
    */
  override def forward(acts: ActivationStack, model: Model): (linalg.Vector[Double], linalg.Vector[Double]) = ???

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
    * @param delta
    * @param acts
    * @param model
    * @return
    */
  override def backward(delta: linalg.Vector[Double], acts: ActivationStack, model: Model): (linalg.Vector[Double], Weight, Bias) = ???

  override val outputSize: Int = 3
  override val inputSize: Int = 4
}
