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

import breeze.linalg.{inv, qr, DenseMatrix => BDM}
import java.util.Random

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD


/**
 * Implementation of the Expectation Maximization algorithm
 * for Probabilistic PCA shown by (Roweis 1997)
 *
 * @param k number of principal components
 */
@Since("2.3")
class PPCA (val k: Int, tol: Double = 1E-4, maxIterations: Int = 100) {
  require(k > 0,
  s"Number of principal components must be positive but got ${k}")

  private implicit def toMLLib : BDM[Double] => Matrix = {
    (m: BDM[Double]) => Matrices.dense(m.rows, m.cols, m.data)
  }

  private implicit def toBDM : Matrix => BDM[Double] = {
    (m: Matrix) => new BDM(m.numRows, m.numCols, m.toArray)
  }

  /**
   * Computes a [[PPCAModel]] that contains the transformation
   * matrix that convert x vectors into the latent space
    * @param sources data NxD
   * @return
   */
  @Since("2.3")
  def fit(sources: RDD[Vector], seed: Int = 1): PPCAModel = {
    val numFeatures = sources.first().size
    require(k <= numFeatures,
      s"source vector size $numFeatures must be no less than k=$k")

    val meanScaler = new StandardScaler().fit(sources).setWithStd(false).setWithMean(true)
    val scaledSources = meanScaler.transform(sources)
    def vectorIndexer: ((Vector, Long)) => IndexedRow = {
      t: (Vector, Long) => new IndexedRow(t._2, t._1)}
    val mat = new IndexedRowMatrix(scaledSources.zipWithIndex().map[IndexedRow](vectorIndexer))
    // val wSeed = Matrices.randn(numFeatures, k, new Random(seed))
    val wSeed = getWSeed(numFeatures, k)
    val wEM = em(wSeed, mat, tol, maxIterations)
    new PPCAModel(k, wEM)
  }

  /**
   * Performs the Expectation Maximization algorithms for
   * Probabilistic PCA
   * @param wSeed initial factor loading matrix
   * @param x mean centered samples
   * @param tol threshold to stop algorithm
   * @param maxIterations max number of iterations if threshold is not reached
   * @return
   */
  @Since("2.3")
  private def em(wSeed: Matrix, x: IndexedRowMatrix, tol: Double, maxIterations: Int): Matrix = {
    def emStep(wOld: Matrix, iteration: Int): Matrix = {
      if (iteration == 0) {
        wOld
      } else {
        val zTr = getZTr(wOld, x) // E-step
        val wNew = getW(zTr, x) // M-step
        val xReconstructed = zTr.multiply(wNew.transpose).toBlockMatrix()
        if (mse(x.toBlockMatrix(), xReconstructed) < tol) {
          wNew
        } else {
          emStep(wNew, iteration - 1)
        }
      }
    }
    val w = emStep(wSeed, maxIterations) // First E-step to initialize
    Matrices.dense(w.numRows, w.numCols, qr(toBDM(w)).q.data.slice(0, w.numCols*w.numRows)) // Q1
  }

  /**
   * Create initial directions over the first d axes
   * (1, 0, 0)
   * (0, 1, 0)
   * (0, 0, 1)
   * (0, 0, 0)
   * (0, 0, 0)
   * @param d number of features
   * @param k number of dimensions of latent space
   * @return
   */
  @Since("2.3")
  private def getWSeed(d: Int, k: Int): Matrix = {
    require(k <= d, s"k=${k} is greater than number of features d=${d}")
    Matrices.dense(d, k, Matrices.eye(d).data.slice(0, k * d))
  }

  @Since("2.3")
  private def mse(mat1: BlockMatrix,
                                   mat2: BlockMatrix): Double = {
    val blkSumSqr = (blk: ((Int, Int), Matrix)) => blk._2.toArray.map(math.pow(_, 2)).sum
    val subBlocks = mat1.subtract(mat2).blocks
    val totalSum = subBlocks.map(blkSumSqr).sum()
    math.sqrt(totalSum)
  }

  /**
   * Computes the projection (low-dimensional representation)
   *
   * Let Z = inv(W'W)W'X'
   * Z' = X(inv(W'W)W')' = X(W inv(W'W)')
   * We compute Z' since X can be huge and only it can multiply in
   * the left as Distributed Matrix
   * @param w factor loading matrix
   * @param x mean centered samples
   * @return
   */
  @Since("2.3")
  private def getZTr(w: Matrix, x: IndexedRowMatrix): IndexedRowMatrix = {
    val wBrz = toBDM(w)
    // alpha W = inc(W'W)'
    val alpha = wBrz * inv(wBrz.t * wBrz).t
    x.multiply(alpha)
  }

  /**
   * Computes the factor loading matrix
   * W = X'Z'inv(ZZ')
   * @param zTr projection of data in low dimensional space
   * @param x mean centered samples
   * @return
   */
  private def getW(zTr: IndexedRowMatrix, x: IndexedRowMatrix): Matrix = {
    val zzInv = inv(zTr.toBlockMatrix().transpose.multiply(zTr.toBlockMatrix()).toBreeze())
    // beta = Z'inv(ZZ')
    val beta = zTr.multiply(zzInv).toBreeze()
    x.toBlockMatrix().transpose.toIndexedRowMatrix().multiply(beta).toBreeze()
  }
}

/**
 * Model fitted by [[PPCA]] that can project vectors to a lower latent space Z
 * @param k number of dimensions of latent space
 * @param w transformation matrix
 */
@Since("2.3")
class PPCAModel private[spark] (val k: Int, val w: Matrix) extends VectorTransformer {
  /**
   * Applies transformation on a vector.
   *
   * @param vector vector to be transformed.
   * @return transformed vector.
   */
  override def transform(vector: Vector): Vector = {
    null
  }

  @Since("2.3")
  def getW: Matrix = w
}
