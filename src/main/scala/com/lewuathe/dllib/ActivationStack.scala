package com.lewuathe.dllib

import scala.collection.mutable.Stack

import breeze.linalg.Vector

/**
  * ActivationStack collects all raw output and activations of each layer.
  * This is a stack listing (u, z)
  * u: raw output that multiplication weight and input
  * z: f(u) assuming f is an activation function
  */
class ActivationStack extends Stack[(Vector[Double], Vector[Double])] {}

