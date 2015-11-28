package com.lewuathe.dllib

import org.apache.spark.mllib.linalg.Vector

private [dllib] case class Instance(label: Double, weight: Double, features: Vector)
}
