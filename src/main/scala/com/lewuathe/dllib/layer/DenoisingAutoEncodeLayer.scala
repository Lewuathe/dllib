package com.lewuathe.dllib.layer

import com.lewuathe.dllib.util.genId

class DenoisingAutoEncodeLayer(override val outputSize: Int,
                              override val inputSize: Int) extends Layer {
  override val id = genId()


}
