import breeze.linalg.DenseVector
import org.scalatest.FlatSpec
import fortytwo.activations._
/**
 * Created by kaisasak on 6/29/14.
 */
class ActivationSpec extends FlatSpec {
  "Sigmoid function" should "return 0.5 at 0.0" in {
    assert(sigmoid(0.0) == 0.5)
  }

  "Sigmoid function for vector" should "return 0.0, 0.5, 1.0 with -100.0, 0.0, 100.0 vector " in {
    val ret = sigmoid(DenseVector(-100.0, 0.0, 100.0))
    assert(ret(0) < 0.0001)
    assert(ret(1) == 0.5)
    assert(ret(2) > 0.9999)
  }
}
