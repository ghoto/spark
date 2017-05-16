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

import breeze.linalg.{eig, inv, norm, svd, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM}
import breeze.linalg.svd.SVD
import java.util.Random

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD


/**
 * Implementation of the Expectation Maximization algorithm
 * for Probabilistic PCA shown by (Roweis 1997)
 *
 * @param k number of principal components
 */
@Since("2.3")
class PPCA (val k: Int, tol: Double = 1E-4, maxIterations: Int = 25, sensible: Boolean = true) {
  require(k > 0,
  s"Number of principal components must be positive but got ${k}")

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
    val wSeed = Matrices.randn(numFeatures, k, new Random(seed))
//    val wSeed = getWSeed(numFeatures, k)
    val (uq, sq) = em(wSeed, mat, tol, maxIterations)
    new PPCAModel(k, uq)
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
  private def em(wSeed: Matrix,
                 x: IndexedRowMatrix,
                 tol: Double,
                 maxIterations: Int): (Matrix, Vector) = {
    val d = wSeed.numRows
    // Compute trace(X'X) for ss computation and RDD[X] for second trace
    val (xRdd, xTxTrace) = if (sensible) {
      val r = x.rows.map((iV) => (iV.index, iV.vector))
      (Some(r), Some(r.map((t: (Long, Vector)) => math.pow(norm(t._2.asBreeze), 2)).sum()))
    } else {
      (None, None)
    }
    def emStep(wOld: Matrix, ssOld: Double, iteration: Int): (Matrix, Vector) = {
      // e-step
      val (mu, sigma) = expectation(wOld, x, ssOld)
      // m-step
      val wNew = newFactorLoadings(mu, x, sigma)
      val ssNew = if (sensible) {
        computeSs(mu, wOld, wNew, xRdd.get, xTxTrace.get)
      } else {
        0.0
      }
      // compute convergence
      //
      // The algorithm converges to
      // wML = Uq(Sq - ssNew*I)^0.5 R
      // Verify that wML has converged
      val diff: BDM[Double] = (wNew.asBreeze - wOld.asBreeze).toDenseMatrix
      val conv = math.sqrt(diff.toArray.par.map((c) => math.pow(c, 2)).sum)
      val SVD(u: BDM[Double], s: BDV[Double], vT) = svd(wNew.asBreeze.toDenseMatrix)
      if (conv < tol || iteration <= 1) {
        val SVD(u: BDM[Double], s: BDV[Double], _) = svd(wNew.asBreeze.toDenseMatrix)
        (Matrices.dense(d, k, u.toArray.slice(0, d * k)),
          Vectors.dense(s.map(_ - math.sqrt(ssNew)).data.slice(0, k)))
      } else {
        emStep(wNew, ssNew, iteration - 1)
      }
    }
    emStep(wSeed, 0, maxIterations)
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
  private def expectation(w: Matrix,
                          x: IndexedRowMatrix,
                          ss: Double = 0.0): (IndexedRowMatrix, BM[Double]) = {
    val n = x.numRows().toDouble
    val d = x.numCols().toInt
    val kd = 1.0/(k * d)
    val wBrz = w.asBreeze.toDenseMatrix
    // Equivalent to
    // val alpha = inv(BDM.eye[Double](d) * ss + wBrz * wBrz.t) * wBrz
    // But operations are kxk instead of dxd
    val alpha = if (ss > 0) {
      wBrz * inv(BDM.eye[Double](w.numCols) + wBrz.t * wBrz/ss) * kd
    } else {
      wBrz * inv(wBrz.t * wBrz) * kd
    }
    val mu = x.multiply(Matrices.fromBreeze(alpha))
    val sigma = if (ss > 0) {
      BDM.eye[Double](w.numCols) * n -
        alpha.t * wBrz * n +
        mu.computeGramianMatrix().asBreeze.toDenseMatrix
    } else {
      mu.computeGramianMatrix().asBreeze.toDenseMatrix
    }
    (mu, sigma)
  }

  private def computeSs(mu: IndexedRowMatrix,
                       wOld: Matrix,
                       wNew: Matrix,
                       xRdd: RDD[(Long, Vector)],
                       xTxTrace: Double): Double = {
    // ssNew = trace[X'X - W mhu'X]/nd
    // ssNew = (trace[X'X] - trace[X'mhuW'])/nd
    // ssNew = (tr1 - tr2)/nd
    //
    // Note that traces apply only in diagonal
    // therefore there isn't needed to compute the
    // whole matrix multiplications
    val muWt = mu.multiply(wOld.transpose)
    val muWtRdd = muWt.rows.map((iV) => (iV.index, iV.vector))
    val xMuWtRdd = xRdd.join(muWtRdd)
    val tr2 = xMuWtRdd.map((t) => t._2._1.asBreeze dot t._2._2.asBreeze).sum()
    (xTxTrace - tr2)/(mu.numRows() * wNew.numCols)
  }

  /**
   * Computes the factor loading matrix
   * W = X'Z inv(Z'Z)
   * @param x mean centered samples
   * @return
   */
  private def newFactorLoadings(mu: IndexedRowMatrix,
                           x: IndexedRowMatrix,
                           sigma: BM[Double]): Matrix = {
    // Compute new loading factor
    val sigmaInv = inv(sigma.toDenseMatrix)
    val beta = mu.multiply(Matrices.fromBreeze(sigmaInv))
    x.toBlockMatrix().transpose.multiply(beta.toBlockMatrix()).toLocalMatrix()
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
