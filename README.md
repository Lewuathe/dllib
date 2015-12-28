dllib [![Build Status](https://travis-ci.org/Lewuathe/dllib.svg?branch=master)](https://travis-ci.org/Lewuathe/dllib) 
================

This is deep learning module running on Apache Spark. Powered by Scala programming language.

# How to use

This is an uploaded on [Spark package](http://spark-packages.org/)

```bash

$ $SPARK_HOME/bin/spark-shell --packages com.lewuathe:dllib_2.11:0.0.2 

```

## Example

```scala

val nn3Form = new Form(Array(
  new FullConnectedLayer(100, 784),
  new ClassificationLayer(10, 100)
))

val nn3Model = Model(nn3Form)
val nn3 = Network(nn3Model, nn3Form)

val multilayerPerceptron = new MultiLayerPerceptron("MNIST", nn3)
val model = multilayerPerceptron.fit(df)

val result = model.transform(df)
```

# License

[Apache v2](http://www.apache.org/licenses/LICENSE-2.0)

# Author

* Kai Sasaki([@Lewuathe](https://github.com/Lewuathe))
