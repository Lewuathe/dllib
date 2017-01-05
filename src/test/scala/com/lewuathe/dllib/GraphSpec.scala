/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.lewuathe.dllib

import org.scalamock.scalatest.MockFactory
import org.scalatest._

import com.lewuathe.dllib.graph.Graph

class GraphSpec extends FlatSpec with Matchers with MockFactory {
  def mockLayers() = {
    val mockLayer23 = mock[MockLayer23]
    val mockLayer34 = mock[MockLayer34]
    Array(mockLayer34, mockLayer23)
  }

  "Form" should "create correct layers" in {
    val graph = new Graph(mockLayers)
    graph.toString should be("id: layer34, 4 -> 3 ==> id: layer23, 3 -> 2")
  }
}
