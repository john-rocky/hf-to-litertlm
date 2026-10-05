// SPDX-License-Identifier: Apache-2.0
package com.kev.snippet

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

private const val PAD = 248044 // <|endoftext|>, right padding
private const val STATE = 248060
private const val QUESTION = 248061
private const val OPTION_END = 248050
private const val DECIDE = 248062

private fun gpu(precision: CompiledModel.GpuOptions.Precision, constantTensorSharing: Boolean? = null) =
  CompiledModel.Options(Accelerator.GPU).apply {
    gpuOptions = CompiledModel.GpuOptions(constantTensorSharing = constantTensorSharing, precision = precision)
  }

/** The rows the pointer head reads: [decide (the last token), each option's 248050], d floats each. */
private fun readoutRows(hidden: FloatArray, length: Int, last: Int, optionEnds: IntArray): List<FloatArray> {
  val d = hidden.size / length
  return (listOf(last) + optionEnds.toList()).map { hidden.copyOfRange(it * d, (it + 1) * d) }
}

/**
 * One Kev row-prefill graph (`*_rowprefill_L{64,128,256,512,1024,2048}_fp16fc_i8emb.tflite`) on the GPU:
 * ids int32 [1, L] + valid float32 [1, L] -> hidden float32 [1, L, d] (d = 1024 for Kev-0.8B, 2560 for Kev-4B).
 * precision: FP16_WITH_FP32_ACCUM (float16 storage, float32 accumulation) or FP32. The default precision (plain float16
 * activations) moves the probabilities outside the parity tolerance. Kev-4B has not run on a phone: on a 12 GB Galaxy
 * S26 both tries, at FP32 and at FP16_WITH_FP32_ACCUM, used up the memory while the graph compiled.
 */
class KevRowGraph(
  file: File,
  private val length: Int,
  env: Environment,
  precision: CompiledModel.GpuOptions.Precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM,
) : AutoCloseable {
  private val model = CompiledModel.create(file.absolutePath, gpu(precision), env)
  private val inputs = listOf("ids", "valid").associateWith { model.createInputBuffer(it, "serving_default") }
  private val outputs = mapOf("hidden" to model.createOutputBuffer("hidden", "serving_default"))

  /**
   * row = [248060] + state + [248061] + instructions + for each option ([248049] + option + [248050]) + [248062], with
   * the Kev repository's tokenizer.json and no special tokens added; optionEnds = the index of each option's 248050.
   */
  fun readout(row: IntArray, optionEnds: IntArray): List<FloatArray> {
    require(row.size <= length && row.last() == DECIDE) { "a row ends with 248062 and fits $length tokens" }
    require(optionEnds.all { row[it] == OPTION_END }) { "optionEnds must point at 248050 tokens" }
    inputs.getValue("ids").writeInt(IntArray(length) { if (it < row.size) row[it] else PAD })
    inputs.getValue("valid").writeFloat(FloatArray(length) { if (it < row.size) 1f else 0f })
    model.run(inputs, outputs, "serving_default")
    return readoutRows(outputs.getValue("hidden").readFloat(), length, row.size - 1, optionEnds)
  }

  override fun close() {
    (inputs.values + outputs.values).forEach { it.close() }
    model.close()
  }
}

/**
 * One Kev shared-state pair (`*_sharedstate_Ls{Ls}_Lq{Lq}_fp16fc_i8emb.tflite`) on the GPU: `state_prefill_<Ls>` runs
 * the request's state once, then `question_step_<Ls>_<Lq>` runs each question from it. state_prefill's output buffers go
 * to question_step as its inputs (no copy through the app). constantTensorSharing (default true) keeps one copy of the
 * weights on the GPU for the two signatures. Without it the pair is faster, but the GPU holds the weights twice: on a
 * Galaxy S26 (12 GB) at FP16_WITH_FP32_ACCUM, Kev-0.8B's Ls 128 pair answered a 5-question request in 430.9 ms without
 * sharing and 624.8 ms with it (median of the requests timed with the GPU clock not capped), and the phone's smallest
 * MemAvailable in the Ls 128 and Ls 256 runs (gate and timing) was 2.7 to 3.1 GB without sharing and 5.6 to 6.1 GB
 * with it.
 * layers = 24 for Kev-0.8B, 32 for Kev-4B: every fourth layer (3, 7, ...) is an attention layer (state k_<l>, v_<l>),
 * the others are Gated DeltaNet layers (gdn_state_<l>, conv_tail_<l>).
 */
class KevPairGraph(
  file: File,
  private val ls: Int,
  private val lq: Int,
  env: Environment,
  layers: Int = 24,
  precision: CompiledModel.GpuOptions.Precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM,
  constantTensorSharing: Boolean = true,
) : AutoCloseable {
  private val model =
    CompiledModel.create(file.absolutePath, gpu(precision, if (constantTensorSharing) true else null), env)
  private val sigState = "state_prefill_$ls"
  private val sigQuestion = "question_step_${ls}_$lq"
  private val stateNames =
    (0 until layers).flatMap { if (it % 4 == 3) listOf("k_$it", "v_$it") else listOf("gdn_state_$it", "conv_tail_$it") }
  private val stateIn = listOf("ids", "valid").associateWith { model.createInputBuffer(it, sigState) }
  private val stateOut = stateNames.associateWith { model.createOutputBuffer(it, sigState) }
  private val questionIn = listOf("ids", "valid", "state_valid").associateWith { model.createInputBuffer(it, sigQuestion) }
  private val questionOut = mapOf("hidden" to model.createOutputBuffer("hidden", sigQuestion))
  private val questionInputs: Map<String, TensorBuffer> = questionIn + stateOut

  /** state = [248060] + the state's tokens, at most Ls. Run once per request, before its questions. */
  fun runState(state: IntArray) {
    require(state.size <= ls && state.first() == STATE) { "a state starts with 248060 and fits $ls tokens" }
    val valid = FloatArray(ls) { if (it < state.size) 1f else 0f }
    stateIn.getValue("ids").writeInt(IntArray(ls) { if (it < state.size) state[it] else PAD })
    stateIn.getValue("valid").writeFloat(valid)
    model.run(stateIn, stateOut, sigState)
    questionIn.getValue("state_valid").writeFloat(valid)
  }

  /**
   * branch = [248061] + instructions + for each option ([248049] + option + [248050]) + [248062], at most Lq tokens: the
   * row without the state. optionEnds index the branch. Returns the same rows as KevRowGraph.readout on the whole row,
   * up to float rounding.
   */
  fun readout(branch: IntArray, optionEnds: IntArray): List<FloatArray> {
    require(branch.size <= lq && branch.first() == QUESTION && branch.last() == DECIDE) {
      "a question starts with 248061, ends with 248062 and fits $lq tokens"
    }
    require(optionEnds.all { branch[it] == OPTION_END }) { "optionEnds must point at 248050 tokens" }
    questionIn.getValue("ids").writeInt(IntArray(lq) { if (it < branch.size) branch[it] else PAD })
    questionIn.getValue("valid").writeFloat(FloatArray(lq) { if (it < branch.size) 1f else 0f })
    model.run(questionInputs, questionOut, sigQuestion)
    return readoutRows(questionOut.getValue("hidden").readFloat(), lq, branch.size - 1, optionEnds)
  }

  override fun close() {
    (stateIn.values + stateOut.values + questionIn.values + questionOut.values).forEach { it.close() }
    model.close()
  }
}
