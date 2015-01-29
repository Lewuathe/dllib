import com.lewuathe.neurallib.activations._
import com.lewuathe.neurallib.networks.NN3
import org.scalatest.FlatSpec
import breeze.linalg._

/**
 * Created by kaisasak on 6/30/14.
 */
class NN3Spec extends FlatSpec{
  "NN3" should "return network output" in {
    val n = NN3(Array(4, 3, 2))
    val x = Vector(0.1, -0.1, 0.1, -0.1)
    assert(x(0) < 100.0 && x(1) < 100.0)
  }

  ignore should "train with given data" in {
    val n = NN3(Array(4, 3, 2))
    val xs = DenseMatrix(
      (0.0, 0.1, 0.0, 0.9),
      (0.9, 0.1, 0.0, -0.2),
      (-0.1, 0.0, 0.1, 0.9),
      (0.9, 0.0, 0.1, 0.0),
      (-0.1, 0.0, 0.3, 0.9),
      (0.8, 0.2, 0.0, 0.0),
      (-0.1, 0.0, 0.2, 0.9),
      (0.9, 0.1, 0.3, 0.0),
      (-0.1, 0.0, 0.2, 0.8),
      (0.9, 0.3, 0.0, -0.1),
      (0.0, 0.1, 0.3, 0.9),
      (0.9, 0.2, 0.1, -0.2),
      (0.0, 0.3, 0.0, 0.9),
      (0.7, 0.5, 0.3, 0.0),
      (0.0, 0.3, 0.1, 0.9),
      (0.9, 0.4, 0.0, -0.1),
      (0.0, 0.1, 0.3, 0.9),
      (0.8, 0.4, 0.0, 0.1)
    )
    val ys = DenseMatrix(
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0)
    )
    n.tied = false
    n.epochs = 30
    n.train(xs, ys)
    val ans1 = n.predict(DenseVector(0.9, 0.1, 0.1, 0.0))
    assert(ans1(0) > 0.60 && ans1(1) < 0.40)
    val ans2 = n.predict(DenseVector(0.0, 0.0, 0.1, 0.9))
    assert(ans2(0) < 0.40 && ans2(1) > 0.60)
  }

  "NN" should "emulate OR logic" in {
    val nn = NN3(Array(2, 3, 1), 0.3)
    val xs = DenseMatrix(
      (0.0, 0.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (1.0, 1.0)
    )
    val ys = DenseMatrix(
      (0.0),
      (1.0),
      (1.0),
      (1.0)
    )
    nn.train(xs, ys)


  }

}
