# Vendored third-party sources

Source code in this directory is copied from upstream releases and patched
only where an upstream build breaks against this repository's pinned PyTorch.
It is consumed through `[tool.uv.sources]` path entries in `pyproject.toml`.

## Conventions

- Keep the upstream tree intact apart from the documented patch. Do not
  reformat, relicense, or re-attribute vendored files.
- Upstream `LICENSE` / `AUTHORS` are preserved verbatim. Vendored files keep
  their upstream copyright headers and are deliberately excluded from `ruff`
  and the pre-commit hooks so they do not drift from upstream.
- Do not ship upstream `tests/` or build artifacts (`build/`, `dist/`,
  `*.egg-info`, `*.so`).

## mamba-ssm 2.3.0

| | |
|---|---|
| Upstream | https://github.com/state-spaces/mamba |
| Version | 2.3.0 |
| License | Apache-2.0 |
| Origin | PyPI sdist `mamba_ssm-2.3.0.tar.gz` |
| sdist sha256 | `8294e12125f76021e4e190f4137e84a84935920eeda5d0037a6917524456b303` |

### Patch

`setup.py` hard-codes `-std=c++17` in `extra_compile_args` for both the HIP
and CUDA branches. PyTorch 2.14's headers require C++20 and fail to compile
under C++17. `torch.utils.cpp_extension` would append `-std=c++20` itself, but
only when no `-std=` flag is already present (`append_std17_if_no_std_present`),
and it does not read `CXXFLAGS`, so there is no environment-variable
workaround.

The patch rewrites the four `-std=c++17` occurrences to `-std=c++20`:

```diff
         extra_compile_args = {
-            "cxx": ["-O3", "-std=c++17"],
+            "cxx": ["-O3", "-std=c++20"],
             "nvcc": append_nvcc_threads(
                 [
                     "-O3",
-                    "-std=c++17",
+                    "-std=c++20",
```

No other file differs from the upstream sdist.

### Why vendored rather than a pinned registry sdist

There is no way to inject a source patch into a registry dependency through
`[tool.uv.sources]`, so the patched tree is carried in-repo and referenced by
path.

### Removal condition

Delete this directory and the matching `[tool.uv.sources]` entry once
mamba-ssm ships a release whose `setup.py` no longer hard-codes `-std=c++17`
(either selecting `-std=c++20` or omitting `-std=` and deferring to PyTorch).
As of 2.3.2.post1 the hard-coded `-std=c++17` is still present, so the fix is
not yet upstream.

## causal-conv1d — deliberately not vendored

`causal-conv1d` is left on its hash-pinned PyPI sdist because it needs no
patch: its `setup.py` CUDA branch passes `"cxx": ["-O3"]` with no `-std=` flag,
so `torch.utils.cpp_extension` appends `-std=c++20` on its own. Only the HIP
branch hard-codes `-std=c++17`, and HIP is not built here.

Verified by building `causal_conv1d-1.6.0` from the pristine sdist against
PyTorch 2.14.0+cu130; the emitted compile line ends in `-std=c++20`.
