// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.d1omni.gate

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The layout scripts/s26_score.py assumes and host/d1_host.py build_inputs makes: [prefix rows | ids | pad] in a bucket
 * of L positions, media / pad / keep_right from P and n, the question type one-hot, the six input shapes, and
 * sel = K scores per row at P + markers[k], little-endian float32.
 */
class RowCodecTest {
  private val pad = 0 // d1-omni's <|pad|>

  private fun refused(f: () -> Unit): Boolean =
    try {
      f()
      false
    } catch (e: IllegalArgumentException) {
      true
    }

  @Test
  fun textIdsStartAtZeroAndArePaddedOnTheRight() {
    // P = 0: build_inputs puts the ids at 0 .. n-1 and the pad id (0) after them
    assertArrayEquals(intArrayOf(1, 17, 21, 16, pad, pad, pad), RowCodec.ids(intArrayOf(1, 17, 21, 16), 0, 7, pad))
  }

  @Test
  fun mediaRowIdsStartAfterThePrefix() {
    // P = 3: positions 0 .. 2 belong to the prefix (the pad id there, as build_inputs' zeros), ids at 3 .. 3+n-1
    assertArrayEquals(intArrayOf(pad, pad, pad, 1, 18, 21, pad, pad), RowCodec.ids(intArrayOf(1, 18, 21), 3, 8, pad))
  }

  @Test
  fun mediaPadAndKeepRightFollowBuildInputs() {
    // P = 3, n = 3, L = 8 (host/d1_host.py: media = t < P, pad = t < P + n, keep_right = t != P - 1)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f), RowCodec.media(3, 8), 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f, 1f, 1f, 0f, 0f), RowCodec.pad(3, 3, 8), 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f), RowCodec.keepRight(3, 8), 0f)
  }

  @Test
  fun textRowHasNoMediaAndKeepsEveryRightTap() {
    // P = 0: media all 0, keep_right all 1 (t != -1 everywhere), pad = the n real ids
    assertArrayEquals(FloatArray(6), RowCodec.media(0, 6), 0f)
    assertArrayEquals(FloatArray(6) { 1f }, RowCodec.keepRight(0, 6), 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f, 0f, 0f), RowCodec.pad(0, 4, 6), 0f)
    assertArrayEquals(FloatArray(6) { 1f }, RowCodec.pad(0, 6, 6), 0f)
  }

  @Test
  fun qtypeIsAOneHotOfThree() {
    assertArrayEquals(floatArrayOf(1f, 0f, 0f), RowCodec.qtypeOneHot(0), 0f) // choice
    assertArrayEquals(floatArrayOf(0f, 1f, 0f), RowCodec.qtypeOneHot(1), 0f) // score
    assertArrayEquals(floatArrayOf(0f, 0f, 1f), RowCodec.qtypeOneHot(2), 0f) // noul
    assertEquals(true, refused { RowCodec.qtypeOneHot(3) })
    assertEquals(true, refused { RowCodec.qtypeOneHot(-1) })
  }

  @Test
  fun prefixRowsComeFirstThenZeros() {
    val width = 4
    val rows = FloatArray(2 * width) { it + 0.5f } // P = 2 rows of D = 4
    val p = RowCodec.prefix(rows, 2, width, 5)
    assertEquals(5 * width, p.size)
    for (i in 0 until 2 * width) assertEquals(i + 0.5f, p[i], 0f)
    for (i in 2 * width until 5 * width) assertEquals(0f, p[i], 0f)
    // a text row: the whole [L, D] prefix is zero
    assertArrayEquals(FloatArray(3 * width), RowCodec.prefix(null, 0, width, 3), 0f)
    // the values must be exactly P x D
    assertEquals(true, refused { RowCodec.prefix(FloatArray(7), 2, width, 5) })
    assertEquals(true, refused { RowCodec.prefix(null, 2, width, 5) })
  }

  @Test
  fun prefixFileIsLittleEndianRowMajor() {
    val width = 3
    val values = floatArrayOf(1f, -2.5f, 3.25f, 4f, 5f, -6.125f) // P = 2
    val bytes = RowCodec.littleEndian(values)
    assertArrayEquals(byteArrayOf(0x00, 0x00, 0x80.toByte(), 0x3F), RowCodec.littleEndian(floatArrayOf(1f)))
    assertArrayEquals(values, RowCodec.prefixFromBytes(bytes, 2, width), 0f)
    // a file whose size is not P x D x 4 is refused (wrong P, wrong D, a truncated transfer)
    assertEquals(true, refused { RowCodec.prefixFromBytes(bytes, 3, width) })
    assertEquals(true, refused { RowCodec.prefixFromBytes(bytes.copyOf(bytes.size - 1), 2, width) })
  }

  @Test
  fun selIsTheScoresAtPrefixPlusMarkersInOptionOrder() {
    // scores of one row of L = 10, value = 10 * position + 0.25; P = 2, markers [1, 4, 6, 7] but K = 3
    val scores = FloatArray(10) { it * 10f + 0.25f }
    val sel = RowCodec.select(scores, 2, intArrayOf(1, 4, 6, 7), 3, 7)
    assertArrayEquals(floatArrayOf(30.25f, 60.25f, 80.25f), sel, 0f)
    // two rows in one stream: K floats per row, rows in file order
    val second = RowCodec.select(scores, 0, intArrayOf(5, 2), 2, 6)
    val stream = RowCodec.littleEndian(sel) + RowCodec.littleEndian(second)
    assertEquals((3 + 2) * 4, stream.size)
    val floats = ByteBuffer.wrap(stream).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val back = FloatArray(5).also { floats.get(it) }
    assertArrayEquals(floatArrayOf(30.25f, 60.25f, 80.25f, 50.25f, 20.25f), back, 0f)
  }

  @Test
  fun markersOutsideTheRowAreRefused() {
    val scores = FloatArray(8)
    assertEquals(true, refused { RowCodec.select(scores, 0, intArrayOf(1, 5), 2, 5) })   // marker 5 >= n = 5
    assertEquals(true, refused { RowCodec.select(scores, 0, intArrayOf(1), 2, 5) })      // K = 2 but one marker
    assertEquals(true, refused { RowCodec.select(scores, 4, intArrayOf(1, 4), 2, 5) })   // P + 4 = 8 outside L = 8
  }

  @Test
  fun rowsThatDoNotFitTheBucketAreRefused() {
    assertEquals(true, refused { RowCodec.ids(IntArray(9), 0, 8, pad) })
    assertEquals(true, refused { RowCodec.ids(IntArray(5), 4, 8, pad) })   // P + n = 9 > 8
    assertEquals(true, refused { RowCodec.pad(0, 0, 8) })                  // no id
    RowCodec.requireFits(4, 4, 8)                                         // exactly L is fine
  }

  @Test
  fun inputShapesAreReadAndChecked() {
    assertEquals(256, RowCodec.graphLength(listOf(1, 256)))
    assertEquals(4096, RowCodec.graphLength(listOf(1, 4096)))
    assertEquals(1024, RowCodec.prefixWidth(listOf(1, 256, 1024), 256))
    RowCodec.requireVector(listOf(1, 256), 256, "media")
    RowCodec.requireOneHot(listOf(1, 3))
    assertEquals(256, RowCodec.lengthOfSignature("decide_256"))
    assertNull(RowCodec.lengthOfSignature("serving_default"))
  }

  @Test
  fun shapesTheGraphCannotHaveAreRefused() {
    val bad: List<() -> Unit> =
      listOf(
        { RowCodec.graphLength(listOf(256)) },
        { RowCodec.graphLength(listOf(2, 256)) },
        { RowCodec.graphLength(listOf(1, 256, 1)) },
        { RowCodec.graphLength(listOf(1, 0)) },
        { RowCodec.prefixWidth(listOf(1, 256), 256) },               // rank 2
        { RowCodec.prefixWidth(listOf(1, 512, 1024), 256) },         // another L than ids
        { RowCodec.prefixWidth(listOf(2, 256, 1024), 256) },         // batch 2
        { RowCodec.prefixWidth(listOf(1, 256, 0), 256) },            // no width
        { RowCodec.requireVector(listOf(1, 128), 256, "pad") },      // another L
        { RowCodec.requireVector(listOf(1, 256, 1), 256, "scores") },// rank 3
        { RowCodec.requireOneHot(listOf(1, 4)) },
        { RowCodec.requireOneHot(listOf(3)) },
      )
    for ((i, f) in bad.withIndex()) assertEquals("case $i must be refused", true, refused(f))
  }

  @Test
  fun nonFiniteCountsNanAndInf() {
    val v = floatArrayOf(1f, Float.NaN, 2f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
    assertEquals(3, RowCodec.nonFinite(v, 0, v.size))
    assertEquals(1, RowCodec.nonFinite(v, 0, 2))
  }

  @Test
  fun medianMatchesNumpy() {
    assertEquals(2.0, RowCodec.median(listOf(3.0, 1.0, 2.0)), 0.0)
    assertEquals(2.5, RowCodec.median(listOf(4.0, 1.0, 3.0, 2.0)), 0.0)
  }
}
