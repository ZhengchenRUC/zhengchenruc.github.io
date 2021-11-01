---
layout: post
title:  "Spark-Scala学习(二)"
date:   2021-9-13
tags: Spark Scala GraphX
subtitle: "Spark-Scala学习(二)"
description: 'Spark Scala GraphX'
color: 'rgb(154,133,255)'
cover: '../images/10.png'
---





引入graphx库

```scala
package scala.spark.graphx

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark._
import org.apache.spark.SparkContext._
```

sbt中添加graphx

```scala
name := "Graph Example" //项目名称
  
version := "1.0" //版本号，自己定义

scalaVersion := "2.11.12" //scala版本

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.5" //最后是spark版本号
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.4.5"
```

构建一个grpahx对象，包含三个部分 vertexRDD, edgeRDD, propertyGraph

```scala
//vertexRDD(64-bit, vertex property)
val vertexArray = Array(
    (1L, ("Alice", 28)),
    (2L, ("Bob", 27)),
    (3L, ("Charlie", 65)),
    (4L, ("David", 42)),
    (5L, ("Ed", 55)),
    (6L, ("Fran", 50)))
// the Edge class stores a srcId, a dstId and the edge property
val edgeArray = Array(
    Edge(2L, 1L, 7),
    Edge(2L, 4L, 2),
    Edge(3L, 2L, 4),
    Edge(3L, 6L, 3),
    Edge(4L, 1L, 1),
    Edge(5L, 2L, 2),
    Edge(5L, 3L, 8),
    Edge(5L, 6L, 3))

// construct the following RDDs from the vertexArray and edgeArray variables.
val vertexRDD: RDD[(Long, (String, Int))] = sc.parallelize(vertexArray)
val edgeRDD: RDD[Edge[Int]] = sc.parallelize(edgeArray)

// build a Property Graph
val graph: Graph[(String, Int), Int] = Graph(vertexRDD, edgeRDD)

```

提交运行

```shell
spark-submit --master spark://master:7077 target/scala-2.11/simple-project_2.11-1.0.jar
```

