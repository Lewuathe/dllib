name := "digit-recognizer"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies  ++= Seq(
  // other dependencies here
  "fortytwo" % "fortytwo_2.10" % "1.0",
  "org.scalanlp" %% "breeze" % "0.8.1",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  "org.scalanlp" %% "breeze-natives" % "0.8.1",
  "org.scalatest" % "scalatest_2.10" % "2.0" % "test",
  "com.github.tototoshi" %% "scala-csv" % "0.8.0"
)

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.8-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)