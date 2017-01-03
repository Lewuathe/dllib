package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}
import breeze.stats.distributions.Binomial

import com.lewuathe.dllib.activations.{sigmoid, sigmoidPrime}
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.{Bias, Weight, Model, ActivationStack}
import com.lewuathe.dllib.util.genId

class DenoisingAutoEncodeLayer(override val outputSize: Int,
                              override val inputSize: Int)
  extends PretrainLayer with ShapeValidator with Visualizable {
  override var id = genId()
  // Temporary ID used for storing pretrain parameters on Model

  val corruptionLevel = 0.7

  protected def corrupt(input: Vector[Double]): Vector[Double] = {
    val mask = Vector(Binomial(1, 1.0 - corruptionLevel)
      .sample(input.length).map(_.toDouble): _*)
    mask :* input
  }

  /**
    * Encode the input to hidden layer
    *
    * @param input
    * @param model
    * @param tmpModel
    * @return
    */
  override def encode(input: Vector[Double], model: Model, tmpModel: Model):
      (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val u: Vector[Double] = weight * corrupt(input) + bias
    val z = sigmoid(u)
    (u, z)
  }

  /**
    * Decode hidden layer value to visible layer
    *
    * @param input
    * @param model
    * @param tmpModel
    * @return
    */
  override def decode(input: Vector[Double], model: Model, tmpModel: Model):
      (Vector[Double], Vector[Double]) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    // Make sure to restore a Bias for pretrain visualization layer
    val bias: Vector[Double] = tmpModel.getBias(id).get.value

    // TODO: decode bias should be stored in model
    val u: Vector[Double] = weight.toDenseMatrix.t * input + bias
    val z = sigmoid(u)
    (u, z)
  }

  /**
    * Calculate the error of output layer between label data and prediction.
    *
    * @param label
    * @param prediction
    * @return
    */
  protected def error(label: Vector[Double], prediction: Vector[Double]):
      Vector[Double] = {
    require(label.size == prediction.size)
    val ret = label - prediction
    ret.map({
      case (d: Double) if d.isNaN => 0.0
      case (d: Double) => d
    })
  }

  /**
    * Returns the form for creating tmp model used while pretraining
    * The layer used as prototype for creating tmp model. Only necessary
    * fields are input size, output size and id.
    *
    * @return A new pretrain layer that is reversed output and input.
    *         It is used mainly for keeping bias value while pretraining.
    */
  override def createTmpLayer: PretrainLayer = {
    val tmpLayer = new DenoisingAutoEncodeLayer(inputSize, outputSize)
    tmpLayer.id = this.id
    tmpLayer
  }

  /**
    * Calculate the output corresponding given input.
    * Input is given as a top of ActivationStack.
    * @param acts
    * @param model
    * @return The output tuple of the layer. First value of the tuple
    *         represents the raw output, the second is applied activation
    *         function of the layer.
    */
  override def forward(acts: ActivationStack, model: Model):
      Vector[Double] = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    validateParamShapes(weight, bias)

    val input = acts.top
    require(input.size == inputSize, "Invalid input")

    val u: Vector[Double] = weight * input + bias
    u
  }

  /**
    * Calculate the delta of this iteration. The input of the layer in forward
    * phase can be restored from ActivationStack. It returns the delta of input
    * layer of this layer and the delta of coefficient and intercept parameter.
 *
    * @param delta
    * @param acts
    * @param model
    * @return
    */
  override def backward(delta: Vector[Double], acts: ActivationStack,
                        model: Model): (Vector[Double], Weight, Bias) = {
    val weight: Matrix[Double] = model.getWeight(id).get.value
    val bias: Vector[Double] = model.getBias(id).get.value

    val thisOutput = acts.pop()
    val thisInput = acts.top

    val dWeight: Weight = new Weight(id, outputSize,
      inputSize)(delta.toDenseVector * thisInput.toDenseVector.t)
    val dBias: Bias = new Bias(id, outputSize)(delta)

    validateParamShapes(dWeight.value, dBias.value)

    val d: Vector[Double] = weight.toDenseMatrix.t * delta.toDenseVector
    (d, dWeight, dBias)
  }
}
