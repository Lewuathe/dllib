dllib [![Build Status](https://travis-ci.org/Lewuathe/dllib.svg?branch=master)](https://travis-ci.org/Lewuathe/dllib) [![codecov](https://codecov.io/gh/Lewuathe/dllib/branch/master/graph/badge.svg)](https://codecov.io/gh/Lewuathe/dllib)  [![GitHub tag](https://img.shields.io/github/tag/Lewuathe/dllib.svg)]() [![license](https://img.shields.io/github/license/Lewuathe/dllib.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Maven Central](https://img.shields.io/maven-central/v/com.lewuathe/dllib_2.11.svg)]()
================

dllib is a distributed deep learning framework running on Apache Spark. See more detail in [documentation](http://www.lewuathe.com/dllib/). dllib is designed to be simple and easy to use for Spark users.

Since dllib has completely same interface of MLlib algorithms, libraries in MLlib can be used for feature engineering or transformation.

# How to use

dllib is uploaded on [Spark Packages](http://spark-packages.org/package/Lewuathe/dllib). You can use from `sperk-shell` directly.

```bash

$ ./bin/spark-shell --packages Lewuathe:dllib:0.0.9

```

If you want use jar package to extend, you can write the configuration in your pom.xml.

```
<dependency>
    <groupId>com.lewuathe</groupId>
    <artifactId>dllib_2.11</artifactId>
    <version>0.0.9</version>
</dependency>
```

## Example

This is an example for classification of [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. Full code can be seen [here](https://github.com/Lewuathe/dllib/blob/master/src/main/scala/com/lewuathe/dllib/example/MNISTApp.scala).

```scala
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{AffineLayer, ReLULayer, SoftmaxLayer}
import com.lewuathe.dllib.network.Network

// Define the network structure as calculation graph.
val graph = new Graph(Array(
  new AffineLayer(100, 784),
  new ReLULayer(100, 100),
  new AffineLayer(10, 100),
  new SoftmaxLayer(10, 10)
))

// Model keeps whole network parameters which should be trained.
// Default is in-memory model.
val model = Model(nn3Graph)

val nn3 = Network(model, graph)

// MultilayerPerceptron defines the optimization algorithms and hyper parameters.
val multilayerPerceptron = new MultiLayerPerceptron("MNIST", nn3)

// We can pass Dataset of Spark to the network.
val trainedModel = multilayerPerceptron.fit(df)

val result = trainedModel.transform(df)

result.filter("label = prediction").count()
```

# License

[Apache v2](http://www.apache.org/licenses/LICENSE-2.0)

# Author

* Kai Sasaki([@Lewuathe](https://github.com/Lewuathe))
