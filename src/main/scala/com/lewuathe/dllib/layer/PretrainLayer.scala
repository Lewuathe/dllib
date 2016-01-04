package com.lewuathe.dllib.layer

import com.lewuathe.dllib.{Weight, Bias, ActivationStack, Model}

abstract class PretrainLayer extends Layer {
  /**
    * Pretraining with input data to the layer.
    * We can assume AutoEncoder can inherit this class.
    * It returns the gradient of weight and bias of the layer.
    * @param acts
    * @param model
    */
  def pretrain(acts: ActivationStack, model: Model): (Weight, Bias)
}
