package com.lewuathe.dllib

import scala.collection.mutable.Stack

import breeze.linalg.Vector

/**
  * ActivationStack collects all raw output of each layer.
  * - Raw output that multiplication weight and input
  */
class ActivationStack extends Stack[Vector[Double]] {}

