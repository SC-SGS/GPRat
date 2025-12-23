# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import sys

from spack.package import *

class Gprat(CMakePackage, CudaPackage):#, ROCmPackage):
    """Gaussian Process Regression using Asynchronous Task."""

    homepage = ""
    url = "https://github.com/SC-SGS/GPRat/archive/refs/tags/v0.1.0.tar.gz"
    git = "https://github.com/SC-SGS/GPRat.git"
    maintainers("constracktor")

    license("MIT")

    version("main", branch="main", preferred=True)
    #version("0.3.0", sha256="")
    version("0.2.0", sha256="85bb1ebca89cef63304a889f9c0d57273ab32aea")
    version("0.1.0", sha256="7dd0bdf0eb322e7d7ca0461655459e59077d0464")

    depends_on("cxx", type="build")

    map_cxxstd = lambda cxxstd: "2a" if cxxstd == "20" else cxxstd
    cxxstds = ("11", "14", "17", "20")
    variant(
        "cxxstd",
        default="17",
        values=cxxstds,
        description="Use the specified C++ standard when building.",
    )

    variant(
        "blas",
        default="openblas",
        description="Define CPU BLAS backend.",
        values=("openblas", "mkl"),
        multi=False,
    )

    variant("bindings", default=False, description="Build Python bindings")

    variant("examples", default=False, description="Build examples")

    variant("format", default=False, description="Build formating targets")

    # Build dependencies
    depends_on("git", type="build")
    depends_on("cmake@3.23:", type="build")
    depends_on("hpx@1.10.0: +static malloc=system networking=none max_cpu_count=256")

    # Backend dependecies
    depends_on("intel-oneapi-mkl shared=false", when="blas=mkl")
    depends_on("openblas fortran=false", when="blas=openblas")

    # CUDA
    depends_on("cuda@12.0.1 +allow-unsupported-compilers", when="+cuda")
    depends_on("hpx@1.10.0: +cuda", when="+cuda")

    # ROCm not supported yet
    #depends_on("rocm", when="+rocm")

    # Only ROCm or CUDA maybe be enabled at once
    #conflicts("+rocm", when="+cuda")

    # Conflicts
    conflicts("blas=openblas", when="@0.1.0")
    # Require clang when building with +cuda
    conflicts('+cuda', when='%gcc', msg='CUDA builds of GPRat require clang, not gcc')
    # GPU support requires at least version 0.3.0
    conflicts('+cuda', when='@:0.2.0')

    def cmake_args(self):
        spec, args = self.spec, []

        args += [
            self.define_from_variant("GPRAT_BUILD_BINDINGS", "bindings"),
            self.define_from_variant("GPRAT_ENABLE_EXAMPLES", "examples"),
            self.define_from_variant("GPRAT_ENABLE_FORMAT_TARGETS", "format"),
        ]
        if self.spec.satisfies("+cuda"):
            args += [self.define("GPRAT_WITH_CUDA", "ON")]
            args += [self.define("CMAKE_CUDA_COMPILER", "clang++")]

            cuda_arch_list = spec.variants['cuda_arch'].value
            cuda_arch = cuda_arch_list[0]
            if cuda_arch != 'none':
                args.append('-DCMAKE_CUDA_ARCHITECTURES={0}'.format(cuda_arch))
        else:
            args += [self.define("GPRAT_WITH_CUDA", "OFF")]

        if self.spec.satisfies("blas=mkl"):
            args += [self.define("GPRAT_ENABLE_MKL", "ON")]
        else:
            args += [self.define("GPRAT_ENABLE_MKL", "OFF")]

        # Measuring the durations of implementation steps with APEX drastically reduces the performance as it adds
        # synchronization points between computations. This option should only be enabled for replicating performance
        # measurements and is therefore not intended to be exposed to the user. (Similarly for Cholesky profiling.)
        args += [self.define("GPRAT_APEX_STEPS", "OFF")]
        args += [self.define("GPRAT_APEX_CHOLESKY", "OFF")]

        return args
