/**
  * Created by sasakikai on 11/26/15.
  */
package object util {
  private val ID_SIZE = 16

  private val random = new scala.util.Random

  // Generate a random string of length n from the given alphabet
  private def randomString(alphabet: String)(n: Int): String =
    Stream.continually(random.nextInt(alphabet.size)).map(alphabet).take(n).mkString

  // Generate a random alphabnumeric string of length n
  private def randomAlphanumericString(n: Int) =
    randomString("abcdefghijklmnopqrstuvwxyz0123456789")(n)

  def genId(): String = randomAlphanumericString(ID_SIZE)
}
