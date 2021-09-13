---
layout: post
title:  "Spark-Scala学习(一)"
date:   2021-9-13
tags: Spark Scala SBT
subtitle: "Spark-Scala学习(一)"
description: 'Spark Scala'
color: 'rgb(154,133,255)'
cover: '../images/10.jpg'
---

一、写一个简单的程序

文件目录

```shell
$ find .
.
./simple.sbt
./src
./src/main
./src/main/scala
./src/main/scala/SimpleApp.scala
```

SimpleApp.scala内容

```scala
/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "YOUR_SPARK_HOME/README.md" // 应该是你系统上的某些文件 spark默认读取hdfs文件，如果读取本地文件前面加file:///xxx
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
```

统计一段文字中a和b的出现次数

simple.sbt内容

```scala
name := "Simple Project" //项目名称

version := "1.0" //版本号，自己定义

scalaVersion := "2.11.12" //scala版本

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.5" //最后是spark版本号
```

使用sbt工具编译jar包

```shell
sbt package
```

使用spark-submit命令运行刚刚的程序

```shell
spark-submit --class SimpleApp --master local target/scala-2.11/simple-project_2.11-1.0.jar
```

结果:

```shell
Lines with a: 61, Lines with b: 30
```

二、spark-submit不同模式

| master URL        | 意义                                                         |
| :---------------- | :----------------------------------------------------------- |
| local             | 使用1个worker线程本地运行Spark（即完全没有并行化）           |
| local[K]          | 使用K个worker线程本地运行Spark（最好将K设置为机器的CPU核数） |
| local[*]          | 根据机器的CPU逻辑核数，尽可能多地使用Worker线程              |
| spark://HOST:PORT | 连接到给定的Spark Standalone集群的Master，此端口必须是Master配置的端口，默认为7077 |
| mesos://HOST:PORT | 连接到给定的Mesos集群的Master，此端口必须是Master配置的端口，默认为5050。若Mesos集群使用ZooKeeper，则master URL使用mesos://zk://…… |
| yarn-client       | 以client模式连接到YARN集群，集群位置将通过HADOOP_CONF_DIR环境变量获得 |
| yarn-cluster      | 以cluster模式连接到YARN集群，集群位置将通过HADOOP_CONF_DIR环境变量获得 |

