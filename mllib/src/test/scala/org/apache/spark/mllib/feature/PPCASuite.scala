/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD

class PPCASuite extends SparkFunSuite with MLlibTestSparkContext{

  private val data = Array(
    Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
  )

  private lazy val dataRDD: RDD[Vector] = spark.sparkContext.parallelize(data)

  test("PCA") {
    println(new PCA(2).fit(dataRDD).pc)
  }

  test("PPCA") {
    println(new PCA(2).fit(irisRDD).pc)
    val w = new PPCA(2, maxIterations = 10, sensible = false).fit(irisRDD, seed = 3).getW
    println(w.numRows, w.numCols)
    println(w)
    //println(new PPCA(3).fit(dataRDD, seed = 1).getW)
  }

  // Famous IRIS dataset (Fischer, 1936)
  // https://archive.ics.uci.edu/ml/datasets/Iris
  private val iris = Seq(
    Vectors.dense(5.1, 3.5, 1.4, 0.2),
    Vectors.dense(4.9, 3.0, 1.4, 0.2),
    Vectors.dense(4.7, 3.2, 1.3, 0.2),
    Vectors.dense(4.6, 3.1, 1.5, 0.2),
    Vectors.dense(5.0, 3.6, 1.4, 0.2),
    Vectors.dense(5.4, 3.9, 1.7, 0.4),
    Vectors.dense(4.6, 3.4, 1.4, 0.3),
    Vectors.dense(5.0, 3.4, 1.5, 0.2),
    Vectors.dense(4.4, 2.9, 1.4, 0.2),
    Vectors.dense(4.9, 3.1, 1.5, 0.1),
    Vectors.dense(5.4, 3.7, 1.5, 0.2),
    Vectors.dense(4.8, 3.4, 1.6, 0.2),
    Vectors.dense(4.8, 3.0, 1.4, 0.1),
    Vectors.dense(4.3, 3.0, 1.1, 0.1),
    Vectors.dense(5.8, 4.0, 1.2, 0.2),
    Vectors.dense(5.7, 4.4, 1.5, 0.4),
    Vectors.dense(5.4, 3.9, 1.3, 0.4),
    Vectors.dense(5.1, 3.5, 1.4, 0.3),
    Vectors.dense(5.7, 3.8, 1.7, 0.3),
    Vectors.dense(5.1, 3.8, 1.5, 0.3),
    Vectors.dense(5.4, 3.4, 1.7, 0.2),
    Vectors.dense(5.1, 3.7, 1.5, 0.4),
    Vectors.dense(4.6, 3.6, 1.0, 0.2),
    Vectors.dense(5.1, 3.3, 1.7, 0.5),
    Vectors.dense(4.8, 3.4, 1.9, 0.2),
    Vectors.dense(5.0, 3.0, 1.6, 0.2),
    Vectors.dense(5.0, 3.4, 1.6, 0.4),
    Vectors.dense(5.2, 3.5, 1.5, 0.2),
    Vectors.dense(5.2, 3.4, 1.4, 0.2),
    Vectors.dense(4.7, 3.2, 1.6, 0.2),
    Vectors.dense(4.8, 3.1, 1.6, 0.2),
    Vectors.dense(5.4, 3.4, 1.5, 0.4),
    Vectors.dense(5.2, 4.1, 1.5, 0.1),
    Vectors.dense(5.5, 4.2, 1.4, 0.2),
    Vectors.dense(4.9, 3.1, 1.5, 0.1),
    Vectors.dense(5.0, 3.2, 1.2, 0.2),
    Vectors.dense(5.5, 3.5, 1.3, 0.2),
    Vectors.dense(4.9, 3.1, 1.5, 0.1),
    Vectors.dense(4.4, 3.0, 1.3, 0.2),
    Vectors.dense(5.1, 3.4, 1.5, 0.2),
    Vectors.dense(5.0, 3.5, 1.3, 0.3),
    Vectors.dense(4.5, 2.3, 1.3, 0.3),
    Vectors.dense(4.4, 3.2, 1.3, 0.2),
    Vectors.dense(5.0, 3.5, 1.6, 0.6),
    Vectors.dense(5.1, 3.8, 1.9, 0.4),
    Vectors.dense(4.8, 3.0, 1.4, 0.3),
    Vectors.dense(5.1, 3.8, 1.6, 0.2),
    Vectors.dense(4.6, 3.2, 1.4, 0.2),
    Vectors.dense(5.3, 3.7, 1.5, 0.2),
    Vectors.dense(5.0, 3.3, 1.4, 0.2),
    Vectors.dense(7.0, 3.2, 4.7, 1.4),
    Vectors.dense(6.4, 3.2, 4.5, 1.5),
    Vectors.dense(6.9, 3.1, 4.9, 1.5),
    Vectors.dense(5.5, 2.3, 4.0, 1.3),
    Vectors.dense(6.5, 2.8, 4.6, 1.5),
    Vectors.dense(5.7, 2.8, 4.5, 1.3),
    Vectors.dense(6.3, 3.3, 4.7, 1.6),
    Vectors.dense(4.9, 2.4, 3.3, 1.0),
    Vectors.dense(6.6, 2.9, 4.6, 1.3),
    Vectors.dense(5.2, 2.7, 3.9, 1.4),
    Vectors.dense(5.0, 2.0, 3.5, 1.0),
    Vectors.dense(5.9, 3.0, 4.2, 1.5),
    Vectors.dense(6.0, 2.2, 4.0, 1.0),
    Vectors.dense(6.1, 2.9, 4.7, 1.4),
    Vectors.dense(5.6, 2.9, 3.6, 1.3),
    Vectors.dense(6.7, 3.1, 4.4, 1.4),
    Vectors.dense(5.6, 3.0, 4.5, 1.5),
    Vectors.dense(5.8, 2.7, 4.1, 1.0),
    Vectors.dense(6.2, 2.2, 4.5, 1.5),
    Vectors.dense(5.6, 2.5, 3.9, 1.1),
    Vectors.dense(5.9, 3.2, 4.8, 1.8),
    Vectors.dense(6.1, 2.8, 4.0, 1.3),
    Vectors.dense(6.3, 2.5, 4.9, 1.5),
    Vectors.dense(6.1, 2.8, 4.7, 1.2),
    Vectors.dense(6.4, 2.9, 4.3, 1.3),
    Vectors.dense(6.6, 3.0, 4.4, 1.4),
    Vectors.dense(6.8, 2.8, 4.8, 1.4),
    Vectors.dense(6.7, 3.0, 5.0, 1.7),
    Vectors.dense(6.0, 2.9, 4.5, 1.5),
    Vectors.dense(5.7, 2.6, 3.5, 1.0),
    Vectors.dense(5.5, 2.4, 3.8, 1.1),
    Vectors.dense(5.5, 2.4, 3.7, 1.0),
    Vectors.dense(5.8, 2.7, 3.9, 1.2),
    Vectors.dense(6.0, 2.7, 5.1, 1.6),
    Vectors.dense(5.4, 3.0, 4.5, 1.5),
    Vectors.dense(6.0, 3.4, 4.5, 1.6),
    Vectors.dense(6.7, 3.1, 4.7, 1.5),
    Vectors.dense(6.3, 2.3, 4.4, 1.3),
    Vectors.dense(5.6, 3.0, 4.1, 1.3),
    Vectors.dense(5.5, 2.5, 4.0, 1.3),
    Vectors.dense(5.5, 2.6, 4.4, 1.2),
    Vectors.dense(6.1, 3.0, 4.6, 1.4),
    Vectors.dense(5.8, 2.6, 4.0, 1.2),
    Vectors.dense(5.0, 2.3, 3.3, 1.0),
    Vectors.dense(5.6, 2.7, 4.2, 1.3),
    Vectors.dense(5.7, 3.0, 4.2, 1.2),
    Vectors.dense(5.7, 2.9, 4.2, 1.3),
    Vectors.dense(6.2, 2.9, 4.3, 1.3),
    Vectors.dense(5.1, 2.5, 3.0, 1.1),
    Vectors.dense(5.7, 2.8, 4.1, 1.3),
    Vectors.dense(6.3, 3.3, 6.0, 2.5),
    Vectors.dense(5.8, 2.7, 5.1, 1.9),
    Vectors.dense(7.1, 3.0, 5.9, 2.1),
    Vectors.dense(6.3, 2.9, 5.6, 1.8),
    Vectors.dense(6.5, 3.0, 5.8, 2.2),
    Vectors.dense(7.6, 3.0, 6.6, 2.1),
    Vectors.dense(4.9, 2.5, 4.5, 1.7),
    Vectors.dense(7.3, 2.9, 6.3, 1.8),
    Vectors.dense(6.7, 2.5, 5.8, 1.8),
    Vectors.dense(7.2, 3.6, 6.1, 2.5),
    Vectors.dense(6.5, 3.2, 5.1, 2.0),
    Vectors.dense(6.4, 2.7, 5.3, 1.9),
    Vectors.dense(6.8, 3.0, 5.5, 2.1),
    Vectors.dense(5.7, 2.5, 5.0, 2.0),
    Vectors.dense(5.8, 2.8, 5.1, 2.4),
    Vectors.dense(6.4, 3.2, 5.3, 2.3),
    Vectors.dense(6.5, 3.0, 5.5, 1.8),
    Vectors.dense(7.7, 3.8, 6.7, 2.2),
    Vectors.dense(7.7, 2.6, 6.9, 2.3),
    Vectors.dense(6.0, 2.2, 5.0, 1.5),
    Vectors.dense(6.9, 3.2, 5.7, 2.3),
    Vectors.dense(5.6, 2.8, 4.9, 2.0),
    Vectors.dense(7.7, 2.8, 6.7, 2.0),
    Vectors.dense(6.3, 2.7, 4.9, 1.8),
    Vectors.dense(6.7, 3.3, 5.7, 2.1),
    Vectors.dense(7.2, 3.2, 6.0, 1.8),
    Vectors.dense(6.2, 2.8, 4.8, 1.8),
    Vectors.dense(6.1, 3.0, 4.9, 1.8),
    Vectors.dense(6.4, 2.8, 5.6, 2.1),
    Vectors.dense(7.2, 3.0, 5.8, 1.6),
    Vectors.dense(7.4, 2.8, 6.1, 1.9),
    Vectors.dense(7.9, 3.8, 6.4, 2.0),
    Vectors.dense(6.4, 2.8, 5.6, 2.2),
    Vectors.dense(6.3, 2.8, 5.1, 1.5),
    Vectors.dense(6.1, 2.6, 5.6, 1.4),
    Vectors.dense(7.7, 3.0, 6.1, 2.3),
    Vectors.dense(6.3, 3.4, 5.6, 2.4),
    Vectors.dense(6.4, 3.1, 5.5, 1.8),
    Vectors.dense(6.0, 3.0, 4.8, 1.8),
    Vectors.dense(6.9, 3.1, 5.4, 2.1),
    Vectors.dense(6.7, 3.1, 5.6, 2.4),
    Vectors.dense(6.9, 3.1, 5.1, 2.3),
    Vectors.dense(5.8, 2.7, 5.1, 1.9),
    Vectors.dense(6.8, 3.2, 5.9, 2.3),
    Vectors.dense(6.7, 3.3, 5.7, 2.5),
    Vectors.dense(6.7, 3.0, 5.2, 2.3),
    Vectors.dense(6.3, 2.5, 5.0, 1.9),
    Vectors.dense(6.5, 3.0, 5.2, 2.0),
    Vectors.dense(6.2, 3.4, 5.4, 2.3),
    Vectors.dense(5.9, 3.0, 5.1, 1.8)
  )

  private lazy val irisRDD = spark.sparkContext.parallelize(iris)
}
