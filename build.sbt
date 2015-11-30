organization := "com.lewuathe"

name := "dllib"

version := "0.0.2"


libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.11.2",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "com.github.tototoshi" % "scala-csv_2.10" % "0.8.0"
)

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.8-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

// Scala 2.9.2 is still supported for 0.2.1, but is dropped afterwards.
// Don't use an earlier version of 2.10, you will probably get weird compiler crashes.
scalaVersion := "2.11.7"

sparkVersion := "1.5.2"

sparkComponents += "mllib"

spShortDescription := "Distributed Deep learning module on Spark"

spDescription :=
  """dllib is a distributed deep learning module running on Spark
    |dllib provides configurable interface and scalable performance
    |that fits your deep learning usage.
  """.stripMargin

spHomepage := "https://github.com/Lewuathe/dllib"

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

publishMavenStyle := true

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

pomExtra := (
  <url>https://github.com/Lewuathe/neurallib</url>
  <licenses>
    <license>
      <name>MIT</name>
      <url>http://opensource.org/licenses/MIT</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <scm>
    <url>git@github.com:Lewuathe/neurallib.git</url>
    <connection>scm:git:git@github.com:Lewuathe/neurallib.git</connection>
  </scm>
  <developers>
    <developer>
      <id>lewuathe</id>
      <name>Kai Sasaki</name>
      <url>http://lewuathe.com</url>
    </developer>
  </developers>
)
