# Vendored QuACK provenance

`quack/` is the Python source snapshot of **quack-kernels 0.6.4** from the tested environment,
with three modified files for this experimental Hopper FP8 forward. It is included so the
experiment does not depend on unpublished edits to site-packages or on a moving upstream version.
Upstream project: <https://github.com/Dao-AILab/quack>.
The upstream Git commit of the installed wheel was not recorded; the package version and this
source snapshot's SHA256 manifest are the reproducibility identifiers. Do not infer a commit from the version.

The original copyright notices are retained. See [QUACK_LICENSE](QUACK_LICENSE) (Apache-2.0).
SonicMoE code is supplied by the enclosing repository at base commit
`7396f3e604827d8186c2e16e64b28ee33d3defd0`, under its root [LICENSE](../../../LICENSE).

## Modified files

- `quack/gemm_sm90.py`: per-K=128 FP8 activation/weight scales, FP32 partial accumulation,
  reuse of scale products, concat Gate/Up addressing, and the optional fusion experiment's
  N warp-group split. The final default does not enable scale-prefetch or fused quantization.
- `quack/epilogue/ops.py`: two-dimensional varlen output offset for the optional scale output.
- `quack/epilogue/visit.py`: paired-accumulator statistics prepass for the optional fused quantization.

[quack-fp8.patch](quack-fp8.patch) is the complete diff against the unmodified installed
0.6.4 versions of these files, including the modification notices. Other Python files match
that installed snapshot. From an unmodified QuACK 0.6.4 source directory, the equivalent patch
can be applied with `git apply /path/to/quack-fp8.patch`. The experiment already contains the
patched files; do not apply the patch a second time.

`_bootstrap.py` selects this copy only for the experimental entry point. Normal SonicMoE
installation is not changed to replace its dependency with this fork. Do not import this fork
and the installed QuACK in the same Python process.
