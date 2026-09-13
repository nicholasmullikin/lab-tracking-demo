# Building VIGRA 1.12.4 for the project's uv Python 3.12 environment

The calibration workspace's display-only view aids call
[VIGRA](https://ukoethe.github.io/vigra/) edge and corner operators directly. VIGRA is
**not published on PyPI** (both `vigra` and `vigranumpy` return 404), so it cannot be
declared in `pyproject.toml` or captured in `uv.lock`. It has to be built from source
and installed into the project virtual environment.

Everything here is display-only tooling. VIGRA is never involved in decoding, in any
stored frame, mask, review, proposal, or correction schedule, or in any provenance
hash. If VIGRA is missing the workspace still runs: the VIGRA-backed controls report
themselves as unavailable with this document's path, and brightness, contrast, and
adaptive threshold keep working on NumPy.

## Why the distribution packages do not work

Fedora 44 ships `boost-python3` built against the **system** Python, which is 3.14:

```console
$ ls /usr/lib64/libboost_python*
/usr/lib64/libboost_python314.so
```

Boost.Python is Python-ABI-specific, and this project pins `>=3.12,<3.13` with a
uv-managed CPython 3.12.13. `sudo dnf install boost-python3-devel` therefore does
**not** unblock the build — it installs the Python 3.14 variant, and vigranumpy's
`config/FindVIGRANUMPY_DEPENDENCIES.cmake` looks for `boost_python312`. Boost.Python
must be built against the uv interpreter as well.

## System packages

These were already present on the target Fedora 44 machine and are required:

- `cmake` (4.3.0), `gcc-c++` (16.2.1), `make`, `git`
- `boost-devel`, `hdf5-devel`, `fftw-devel`
- `libjpeg-turbo-devel`, `libpng-devel`, `libtiff-devel`, `openexr-devel`

If any are missing:

```bash
sudo dnf install cmake gcc-c++ make git boost-devel hdf5-devel fftw-devel \
  libjpeg-turbo-devel libpng-devel libtiff-devel openexr-devel
```

`boost-python3-devel` is intentionally **not** in that list; see above.

## Pinned versions

| Component | Pin |
| --- | --- |
| VIGRA | tag `Version-1-12-4`, commit `de98f930b66d461360a2d5dc8f9adfa84bb01058` |
| Boost | `1.90.0` source tarball from `archives.boost.io` |
| Python | uv-managed CPython `3.12.13` |
| NumPy | `2.5.3` (from `uv.lock`) |

Scratch build root used below: `/home/nick/scratch/vigra-build` — deliberately outside
the repository so no build output can land in `runs/` or `artifacts/`. The install
prefix must stay in place after the build, because the installed extension modules
carry an RPATH pointing at `${SCRATCH}/prefix/lib`.

## 1. Build Boost.Python against the uv interpreter

```bash
SCRATCH=/home/nick/scratch/vigra-build
VENV_PYTHON=/home/nick/src/battle/.venv/bin/python
PY_BASE=$("$VENV_PYTHON" -c 'import sys; print(sys.base_prefix)')

mkdir -p "$SCRATCH" && cd "$SCRATCH"
curl -sL -o boost_1_90_0.tar.bz2 \
  https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.bz2
tar xf boost_1_90_0.tar.bz2
cd boost_1_90_0

cat > user-config.jam <<EOF
using gcc ;
using python : 3.12
  : $VENV_PYTHON
  : $PY_BASE/include/python3.12
  : $PY_BASE/lib
  ;
EOF

./bootstrap.sh --with-libraries=python --prefix="$SCRATCH/prefix" \
  --with-python="$VENV_PYTHON"
./b2 --user-config=user-config.jam --with-python python=3.12 \
  variant=release link=shared threading=multi -j"$(nproc)" \
  --prefix="$SCRATCH/prefix" install
```

This produces `$SCRATCH/prefix/lib/libboost_python312.so`.

## 2. Build and install VIGRA + vigranumpy

```bash
cd "$SCRATCH"
git clone --depth 1 --branch Version-1-12-4 https://github.com/ukoethe/vigra.git src
mkdir -p build && cd build

cmake ../src \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SCRATCH/prefix" \
  -DBOOST_ROOT="$SCRATCH/prefix" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DPython_EXECUTABLE="$VENV_PYTHON" \
  -DPython_ROOT_DIR="$PY_BASE" \
  -DBUILD_DOCS=ON -DBUILD_TESTS=ON \
  -DAUTOBUILD_TESTS=OFF -DAUTOEXEC_TESTS=OFF \
  -DWITH_VIGRANUMPY=ON -DWITH_HDF5=ON -DWITH_OPENEXR=ON \
  -DCMAKE_INSTALL_RPATH="$SCRATCH/prefix/lib" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DVIGRANUMPY_INSTALL_DIR=/home/nick/src/battle/.venv/lib/python3.12/site-packages

make -j"$(nproc)"
make install
```

Notes on the non-obvious flags:

- `BUILD_DOCS=ON` / `BUILD_TESTS=ON` only create the targets. `AUTOBUILD_TESTS=OFF`
  keeps `make` from compiling the test programs. Turning either option off makes
  configuration fail in 1.12.4: `vigranumpy/test/CMakeLists.txt` calls the
  `VIGRA_NATIVE_PATH` macro that `BUILD_TESTS` brings in, and the source-package and
  `doc_python` targets depend on the `doc_cpp` target that `BUILD_DOCS` brings in.
- The three RPATH flags are required. Without them `import vigra` fails with
  `ImportError: libvigraimpex.so.11: cannot open shared object file`, because the
  extension modules live in the venv while their shared libraries live in the prefix.
- `VIGRANUMPY_INSTALL_DIR` puts the `vigra` package straight into the venv, so
  `uv run python -c "import vigra"` works with no `PYTHONPATH` or `LD_LIBRARY_PATH`.

## 3. Verify

```console
$ cd /home/nick/src/battle && uv run python -c "import vigra; print(vigra.version)"
1.12.4
```

`uv run` does not remove the package during its sync check, but a destructive
environment reset (`uv sync --reinstall`, or deleting `.venv`) will. Re-run step 2's
`make install` to restore it; the Boost build in step 1 does not need repeating.

## Which operators come from VIGRA

| Control | Backend |
| --- | --- |
| Brightness, Contrast | NumPy — VIGRA has no such operator |
| Adaptive threshold (mean) | NumPy integral-image box mean — VIGRA has no adaptive threshold |
| Adaptive threshold (gaussian) | NumPy threshold over `vigra.filters.gaussianSmoothing` |
| Canny edges | `vigra.analysis.cannyEdgeImageWithThinning` / `cannyEdgeImage` |
| Zero crossings | `vigra.filters.laplacianOfGaussian` + NumPy sign-change marking |
| Shen–Castan (ISEF) | `vigra.analysis.shenCastanEdgeImage` |
| Boundary tensor energy | `vigra.filters.boundaryTensor2D` |
| Corner response function | `vigra.analysis.cornernessHarris` |
| Beaudet | `vigra.analysis.cornernessBeaudet` |
| Rohr | `vigra.analysis.cornernessRohr` |
| Förstner | `vigra.analysis.cornernessFoerstner` |
| Tensor corner/junction | `vigra.analysis.cornernessBoundaryTensor` |

Zero crossings is the only partial case: vigranumpy 1.12.4 does not export
`vigra::zeroCrossings`, so the Laplacian is VIGRA's while the crossings are marked at
whole-pixel positions in NumPy rather than at VIGRA's sub-pixel ones. The UI labels it
that way.
