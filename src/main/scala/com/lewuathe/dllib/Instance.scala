package com.lewuathe.dllib

import breeze.linalg.Vector

private [dllib] case class Instance(label: Vector[Double],
    weight: Double, features: Vector[Double]) {
}
