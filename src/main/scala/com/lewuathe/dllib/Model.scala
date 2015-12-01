package com.lewuathe.dllib

import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.layer.Layer

class ModelShape(form: Form) {
  val weightShape = form.layers.map({
    case layer: Layer => (layer.id, layer.outputSize, layer.inputSize)
  })
  val biasShape = form.layers.map({
    case layer: Layer => (layer.id, layer.outputSize)
  })
}

class Model(form: Form, isZero: Boolean = false)
           (implicit ws: Map[String, Weight], bs: Map[String, Bias]) {
  val shape: ModelShape = new ModelShape(form)

  def init(): (Map[String, Weight], Map[String, Bias]) = {
    val weights: Map[String, Weight] = form.layers.map({
      case layer: Layer => {
        val w = Weight(layer.id, layer.outputSize, layer.inputSize, isZero)
        (w.id, w)
      }
    }).toMap

    val biases: Map[String, Bias] = form.layers.map({
      case layer: Layer => {
        val b = Bias(layer.id, layer.outputSize, isZero)
        (b.id, b)
      }
    }).toMap
    (weights, biases)
  }

  val (weights, biases) = if (ws == null && bs == null) {
    init()
  } else {
    (ws, bs)
  }

  def +(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w + that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b + that.biases(id))
    })
    new Model(this.form)(newWeights, newBiases)
  }
}

object Model {
  implicit val nullWeight: Map[String, Weight] = null
  implicit val nullBias : Map[String, Bias] = null

  def apply(form: Form): Model = new Model(form)
  def zero(form: Form): Model = new Model(form, isZero = true)
}
