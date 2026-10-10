// SPDX-License-Identifier: Apache-2.0
package com.phocinae.gate

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Half
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only answer gate for the Phocinae-Largha-150M-v1 LiteRT graphs in the laya contract of hfmodels-android
 * (`EncoderDecisions`, family laya): main graph input_ids int32 [1,L] (or inputs_embeds float [1,L,384] looked up from
 * an fp16 token table) + attention_mask float [1,L] + qtype_onehot float [1,3] -> token_logits [1,L], pooled_cls [1,384];
 * the host gathers the option markers, computes the four act features and runs the act head graph (always on the CPU).
 * Reads files/<rows> (ids, markers, qtype per row), runs every row once on the requested accelerator, then times the
 * rows listed under "timing_ids" (5 warm-ups + N timed calls each), and writes files/<report>. The host compares the
 * logits with the publisher's reference.
 */
class GateActivity : Activity() {
  private lateinit var status: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    if (Build.VERSION.SDK_INT >= 27) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    status = TextView(this).apply { text = "Phocinae gate: starting" }
    setContentView(status)
    val x = intent
    val args =
      GateArgs(
        graph = x.getStringExtra("graph") ?: "phocinae_L512_ids_v2_fp16fc_i8emb.tflite",
        act = x.getStringExtra("act") ?: "phocinae_act_head_fp32.tflite",
        accel = x.getStringExtra("accel") ?: "gpu",
        precision = x.getStringExtra("precision") ?: "fp32",
        table = x.getStringExtra("table") ?: "",
        rows = x.getStringExtra("rows") ?: "gate_rows.json",
        report = x.getStringExtra("report") ?: "gate_report.json",
        seq = x.getIntExtra("seq", 512),
        limit = x.getIntExtra("limit", 0),
        timed = x.getIntExtra("timed", 20),
        coolMs = x.getIntExtra("cool_ms", 0),
        threads = x.getIntExtra("threads", 4),
      )
    Thread({ runGate(args) }, "PhocinaeGate").start()
  }

  private data class GateArgs(
    val graph: String,
    val act: String,
    val accel: String,
    val precision: String,
    val table: String,
    val rows: String,
    val report: String,
    val seq: Int,
    val limit: Int,
    val timed: Int,
    val coolMs: Int,
    val threads: Int,
  )

  private class Call(val logits: FloatArray, val act: FloatArray, val finite: Boolean, val nonfiniteReal: Int,
                     val lookupMs: Double, val writeMs: Double, val runMs: Double, val readMs: Double, val actMs: Double, val totalMs: Double)

  private fun show(text: String) {
    Log.i(TAG, text)
    runOnUiThread { status.text = text }
  }

  private fun runGate(args: GateArgs) {
    val out = JSONObject()
    out.put("graph", args.graph).put("act", args.act).put("accel", args.accel).put("precision", args.precision)
    out.put("table", args.table).put("seq", args.seq).put("litert", LITERT_VERSION).put("threads", args.threads)
    out.put("device", Build.MODEL).put("android", Build.VERSION.SDK_INT).put("build", Build.DISPLAY)
    out.put("uptime_s_start", SystemClock.elapsedRealtime() / 1000.0)
    var model: CompiledModel? = null
    var actModel: CompiledModel? = null
    var table: WordTable? = null
    try {
      val dir = filesDir
      val rowsDoc = JSONObject(File(dir, args.rows).readText())
      val rows = rowsDoc.getJSONArray("rows")
      val count = if (args.limit > 0) minOf(args.limit, rows.length()) else rows.length()
      val env =
        Environment.create(
          this,
          mapOf(
            Environment.Option.DispatchLibraryDir to applicationInfo.nativeLibraryDir,
            Environment.Option.CompilerPluginLibraryDir to applicationInfo.nativeLibraryDir,
          ),
        )
      show("compiling ${args.graph} on ${args.accel} ${args.precision}")
      val compileStart = System.nanoTime()
      model = CompiledModel.create(File(dir, args.graph).absolutePath, options(args), env)
      val compileMs = ms(System.nanoTime() - compileStart)
      out.put("compile_ms", compileMs)
      Log.i(TAG, "PHOCINAE_GATE compiled in $compileMs ms")
      val actStart = System.nanoTime()
      actModel = CompiledModel.create(File(dir, args.act).absolutePath, CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(numThreads = 1)
      }, env)
      out.put("act_compile_ms", ms(System.nanoTime() - actStart))
      val hostLookup = args.table.isNotEmpty()
      val tokenInput = if (hostLookup) "inputs_embeds" else "input_ids"
      val inputs = listOf(tokenInput, "attention_mask", "qtype_onehot").associateWith { model.createInputBuffer(it, SIGNATURE) }
      val outputs = listOf("token_logits", "pooled_cls").associateWith { model.createOutputBuffer(it, SIGNATURE) }
      val actInputs = listOf("pooled_cls", "feats").associateWith { actModel.createInputBuffer(it, SIGNATURE) }
      val actOutputs = mapOf("act_logits" to actModel.createOutputBuffer("act_logits", SIGNATURE))
      if (hostLookup) table = WordTable(File(dir, args.table))
      val L = args.seq
      val ids = IntArray(L)
      val embeds = if (hostLookup) FloatArray(L * HIDDEN) else null
      val mask = FloatArray(L)
      val onehot = FloatArray(3)
      val feats = FloatArray(4)

      fun call(row: JSONObject): Call {
        val rowIds = row.getJSONArray("ids")
        val n = rowIds.length()
        require(n <= L) { "row ${row.getString("id")} ($n tokens) exceeds the window $L" }
        val markers = row.getJSONArray("markers")
        val qtype = row.getInt("qtype")
        val t0 = System.nanoTime()
        for (p in 0 until L) {
          ids[p] = if (p < n) rowIds.getInt(p) else PAD_ID
          mask[p] = if (p < n) 1f else 0f
        }
        if (hostLookup) for (p in 0 until L) table!!.copyRow(ids[p], embeds!!, p * HIDDEN)
        onehot.fill(0f); onehot[qtype] = 1f
        val t1 = System.nanoTime()
        if (hostLookup) inputs.getValue("inputs_embeds").writeFloat(embeds!!) else inputs.getValue("input_ids").writeInt(ids)
        inputs.getValue("attention_mask").writeFloat(mask)
        inputs.getValue("qtype_onehot").writeFloat(onehot)
        val t2 = System.nanoTime()
        model.run(inputs, outputs, SIGNATURE)
        val t3 = System.nanoTime()
        val tokenLogits = outputs.getValue("token_logits").readFloat()
        val pooled = outputs.getValue("pooled_cls").readFloat()
        val t4 = System.nanoTime()
        require(tokenLogits.size == L && pooled.size == HIDDEN) { "unexpected output shapes ${tokenLogits.size} ${pooled.size}" }
        var nonfiniteReal = 0
        for (p in 0 until n) if (!tokenLogits[p].isFinite()) nonfiniteReal++
        val raw = FloatArray(markers.length()) { tokenLogits[markers.getInt(it)] }
        val finite = raw.all { it.isFinite() } && pooled.all { it.isFinite() }
        var act = floatArrayOf(Float.NaN, Float.NaN)
        if (finite) {
          actFeatures(raw, feats)
          actInputs.getValue("pooled_cls").writeFloat(pooled)
          actInputs.getValue("feats").writeFloat(feats)
          actModel.run(actInputs, actOutputs, SIGNATURE)
          act = actOutputs.getValue("act_logits").readFloat()
        }
        val t5 = System.nanoTime()
        return Call(raw, act, finite, nonfiniteReal, ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), ms(t4 - t3), ms(t5 - t4), ms(t5 - t1))
      }

      if (args.coolMs > 0) {
        show("cooling ${args.coolMs} ms after the compile")
        Thread.sleep(args.coolMs.toLong())
      }
      val records = JSONArray()
      val totals = ArrayList<Double>()
      val runs = ArrayList<Double>()
      val gateStart = System.nanoTime()
      for (i in 0 until count) {
        val row = rows.getJSONObject(i)
        val c = call(row)
        totals.add(c.totalMs); runs.add(c.runMs)
        records.put(
          JSONObject()
            .put("id", row.getString("id"))
            .put("finite", c.finite)
            .put("nonfinite_real", c.nonfiniteReal)
            .put("logits", JSONArray().also { a -> c.logits.forEach { a.put(if (it.isFinite()) it.toDouble() else JSONObject.NULL) } })
            .put("act_logits", JSONArray().also { a -> c.act.forEach { a.put(if (it.isFinite()) it.toDouble() else JSONObject.NULL) } })
            .put("lookup_ms", c.lookupMs).put("write_ms", c.writeMs).put("run_ms", c.runMs).put("read_ms", c.readMs)
            .put("act_ms", c.actMs).put("total_ms", c.totalMs)
        )
        if (i % 100 == 0) show("row $i / $count: ${"%.2f".format(c.totalMs)} ms")
      }
      out.put("gate_seconds", ms(System.nanoTime() - gateStart) / 1000.0)
      val warm = totals.drop(WARMUP).sorted()
      val warmRun = runs.drop(WARMUP).sorted()
      out.put(
        "summary",
        JSONObject()
          .put("rows", count)
          .put("finite_rows", (0 until records.length()).count { records.getJSONObject(it).getBoolean("finite") })
          .put("warm_median_total_ms", if (warm.isNotEmpty()) warm[warm.size / 2] else JSONObject.NULL)
          .put("warm_median_run_ms", if (warmRun.isNotEmpty()) warmRun[warmRun.size / 2] else JSONObject.NULL)
          .put("warm_min_ms", if (warm.isNotEmpty()) warm.first() else JSONObject.NULL)
          .put("warm_max_ms", if (warm.isNotEmpty()) warm.last() else JSONObject.NULL)
          .put("first_call_ms", if (totals.isNotEmpty()) totals.first() else JSONObject.NULL)
      )
      out.put("rows", records)
      // timing rows: 5 warm-ups then `timed` calls each, medians
      val timingIds = rowsDoc.optJSONArray("timing_ids")
      if (timingIds != null && args.timed > 0) {
        val byId = HashMap<String, JSONObject>()
        for (i in 0 until rows.length()) byId[rows.getJSONObject(i).getString("id")] = rows.getJSONObject(i)
        val timing = JSONArray()
        for (k in 0 until timingIds.length()) {
          val row = byId[timingIds.getString(k)] ?: continue
          show("timing ${row.getString("id")}")
          repeat(WARMUP) { call(row) }
          val samples = (0 until args.timed).map { call(row) }
          fun med(xs: List<Double>) = xs.sorted()[xs.size / 2]
          timing.put(
            JSONObject()
              .put("id", row.getString("id"))
              .put("row_len", row.getJSONArray("ids").length())
              .put("K", row.getJSONArray("markers").length())
              .put("n", args.timed).put("warmups", WARMUP)
              .put("total_ms_median", med(samples.map { it.totalMs })).put("total_ms_min", samples.minOf { it.totalMs }).put("total_ms_max", samples.maxOf { it.totalMs })
              .put("run_ms_median", med(samples.map { it.runMs })).put("run_ms_min", samples.minOf { it.runMs }).put("run_ms_max", samples.maxOf { it.runMs })
              .put("write_ms_median", med(samples.map { it.writeMs })).put("read_ms_median", med(samples.map { it.readMs }))
              .put("act_ms_median", med(samples.map { it.actMs })).put("lookup_ms_median", med(samples.map { it.lookupMs }))
              .put("finite", samples.all { it.finite })
          )
        }
        out.put("timing", timing)
      }
      out.put("status", "DONE")
      (inputs.values + outputs.values + actInputs.values + actOutputs.values).forEach(TensorBuffer::close)
      show("done: warm median ${if (warm.isNotEmpty()) warm[warm.size / 2] else -1.0} ms")
    } catch (failure: Throwable) {
      Log.e(TAG, "PHOCINAE_GATE failed", failure)
      out.put("status", "FAILED").put("error", failure.toString())
      show("failed: $failure")
    } finally {
      runCatching { model?.close() }
      runCatching { actModel?.close() }
      runCatching { table?.close() }
    }
    out.put("uptime_s_end", SystemClock.elapsedRealtime() / 1000.0)
    val tmp = File(filesDir, "${args.report}.partial")
    tmp.writeText(out.toString())
    tmp.renameTo(File(filesDir, args.report))
    Log.i(TAG, "PHOCINAE_GATE report ${args.report} ${out.optString("status")}")
  }

  private fun options(args: GateArgs): CompiledModel.Options =
    when (args.accel) {
      "gpu" ->
        CompiledModel.Options(Accelerator.GPU).apply {
          if (args.precision != "default")
            gpuOptions =
              CompiledModel.GpuOptions(
                precision =
                  when (args.precision) {
                    "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                    "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                    "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                    else -> CompiledModel.GpuOptions.Precision.DEFAULT
                  }
              )
        }
      "npu" ->
        CompiledModel.Options(Accelerator.NPU).apply {
          qualcommOptions = CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST)
        }
      else -> CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = args.threads) }
    }

  /** The four act-head features of the publisher's `_score` / the SDK's LayaDecode.actFeatures: softmax of the raw marker logits. */
  private fun actFeatures(raw: FloatArray, dst: FloatArray) {
    val m = raw.max()
    val e = DoubleArray(raw.size) { exp((raw[it] - m).toDouble()) }
    val s = e.sum()
    val p = DoubleArray(raw.size) { e[it] / s }
    val k = max(p.size, 2)
    val sorted = p.sortedDescending()
    val top1 = sorted[0]
    val top2 = if (sorted.size > 1) sorted[1] else 0.0
    var ent = 0.0
    for (v in p) ent -= v * ln(max(v, 1e-9))
    ent /= ln(k.toDouble())
    dst[0] = top1.toFloat(); dst[1] = (top1 - top2).toFloat(); dst[2] = ent.toFloat(); dst[3] = k.toFloat() / 255f
  }

  /** Memory-mapped [vocab, 384] float16 row-major table (the SDK's TokenTable / laya's token_embeddings_fp16.bin). */
  private class WordTable(file: File) : AutoCloseable {
    private val raf = RandomAccessFile(file, "r")
    private val buffer = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).order(ByteOrder.LITTLE_ENDIAN)
    private val shorts = buffer.asShortBuffer()
    val rows = (file.length() / (HIDDEN * 2)).toInt()

    init { require(file.length() % (HIDDEN * 2) == 0L) { "table size ${file.length()} is not a multiple of a row" } }

    fun copyRow(id: Int, dst: FloatArray, offset: Int) {
      val base = id * HIDDEN
      for (c in 0 until HIDDEN) dst[offset + c] = Half.toFloat(shorts.get(base + c))
    }

    override fun close() = raf.close()
  }

  companion object {
    private const val TAG = "PhocinaeGate"
    private const val LITERT_VERSION = "2.2.0"
    private const val SIGNATURE = "serving_default"
    private const val HIDDEN = 384
    private const val PAD_ID = 0
    private const val WARMUP = 5

    private fun ms(nanos: Long) = nanos / 1_000_000.0
  }
}
