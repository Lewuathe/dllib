package com.lewuathe.dllib

import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.layer.Layer

class ModelShape(form: Form) extends Serializable {
  val weightShape = form.layers.map({
    case layer: Layer => (layer.id, layer.outputSize, layer.inputSize)
  })
  val biasShape = form.layers.map({
    case layer: Layer => (layer.id, layer.outputSize)
  })
}

class Model(form: Form, isZero: Boolean = false)
           (implicit ws: Map[String, Weight], bs: Map[String, Bias]) extends Serializable {
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

  var (weights, biases) = if (ws == null && bs == null) {
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

  def -(that: Model): Model = {
    require(this.weights.size == that.weights.size)
    require(this.biases.size == that.biases.size)
    val newWeights = this.weights.map({
      case (id, w) => (id, w - that.weights(id))
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b - that.biases(id))
    })
    new Model(this.form)(newWeights, newBiases)
  }

  def /(denom: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w / denom)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b / denom)
    })
    new Model(this.form)(newWeights, newBiases)
  }

  def *(times: Double): Model = {
    val newWeights = this.weights.map({
      case (id, w) => (id, w * times)
    })
    val newBiases = this.biases.map({
      case (id, b) => (id, b * times)
    })
    new Model(this.form)(newWeights, newBiases)
  }

  def +(that: Weight): Model = {
    val oldWeight = weights.get(that.id).get
    weights = weights + (that.id -> (oldWeight + that))
    this
  }

  def +(that: Bias): Model = {
    val oldBias = biases.get(that.id).get
    biases = biases + (that.id -> (oldBias + that))
    this
  }

  def getWeight(id: String): Option[Weight] = weights.get(id)
  def getBias(id: String): Option[Bias] = biases.get(id)

  override def toString: String = {
    "Model\n  " +
    "  Weights\n" +
    weights.map({
      case (id, w) => s"    id=>${id}, weight=>${w.value}\n"
    }) +
    "  Biases\n" +
    biases.map({
      case (id, b) => s"    id=>${id}, weight=>${b.value}\n"
    })
  }
}

object Model {
  implicit val nullWeight: Map[String, Weight] = null
  implicit val nullBias : Map[String, Bias] = null

  def apply(form: Form): Model = new Model(form)
  def zero(form: Form): Model = new Model(form, isZero = true)
}
