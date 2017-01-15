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

import breeze.linalg.Vector
import org.scalatest._

import com.lewuathe.dllib.activations.{relu, reluPrime, softplus, softplusPrime}

class ActivationSpec extends FlatSpec with Matchers {
  "Rectified linear unit" should "return given value" in {
    val x = Vector(-2.0, 0.0, 2.0)
    relu(x) should be (Vector(0.0, 0.0, 2.0))
  }

  "Rectified linear unit" should "return 1.0 as differential" in {
    val x = Vector(-2.0, 0.0, 2.0)
    reluPrime(x) should be (Vector(0.0, 0.0, 1.0))
  }

  "Softplus unit" should "return 0.0 with small value" in {
    val x = Vector(-10.0, 0.0, 10.0)
    softplus(x) should be (Vector(4.5398899216870535E-5, 0.6931471805599453, 10.000045398899218))
  }

  "Softplus" should "return differential properly" in {
    val x = Vector(-10.0, 0.0, 10.0)
    softplusPrime(x) should be (Vector(4.5397868702434395E-5, 0.5, 0.9999546021312976))
  }
}
