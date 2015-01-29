import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.FlatSpec
import neurallib.regression.LogisticRegression

/**
 * Created by sasakiumi on 7/24/14.
 */
class LogisticsRegressionSpec extends FlatSpec{
  "LogisticsRegression" should "supervised learning" in {
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
    val lregression = LogisticRegression(4, 2)
    lregression.train(xs, ys)
    val ans = lregression.predict(DenseVector(0.0, 0.1, 0.3, 0.9))
    assert(ans(0) < ans(1))
  }
}
