## compiling lblrtm

LBLRTM v12.2 can be compiled , via TelFit's build. Four things had to be fixed, in order:

1. No system Fortran compiler. /usr/bin/gfortran doesn't exist on this machine (only gcc and make). Solution: conda-forge, which was

conda create -y -p ./.conda-telfit "python=3.9" gfortran_linux-64 gcc_linux-64 numpy scipy matplotlib -c conda-forge


That gives GNU Fortran 16.2.0 as x86_64-conda-linux-gnu-gfortran. No sudo needed.


2. TelFit's setup.py couldn't find it. Its compiler search is hardcoded to bare names:

compilers = ["gfortran", "ifort", "g95"]
...
subprocess.check_call([compiler, '--help'], stdout=open("/dev/null"))

It catches OSError and, finding none, raises OSError("Suitable

It catches OSError and, finding none, raises OSError("Suitable compiler not found!"). Setting FC/F77 does nothing — it never reads them. Fix was a symlink into the conda bin dir:

ln -sf x86_64-conda-linux-gnu-gfortran .conda-telfit/bin/gfortran

3. Missing build dependencies, with a misleading error. pip install TelFit failed with ERROR: No .egg-info directory found — modern pip swallowing the real message. Running python setup.py build directly showed it: You need to install cython before running setup.py. Installed cython<3 plus requests, fortranformat, lockfile, astropy, pysynphot (with --no-build-isolation).

4. Modern gfortran rejects the old F77. This one TelFit handles itself — it has a gfortran_mode() function that patches lines 229 and 342 of lblrtm/build/makefile.common to inject -std=legacy into FCFLAG, precisely because gfortran ≥ 7.5 is stricter. That fired automatically once the compiler was found.

The build then pulled ~1.9 GB into ~/.TelFit/ (LBLRTM v12.2.1, LNFL v2.6, the AER v3.2 line file) and produced
lblrtm_v12.2_linux_gnu_sgl and lnfl_v2.6_linux_gnu_sgl.

One more gotcha after the build, not a compilation issue but it blocks use: import telfit dies on from scipy import mat in TelluricFitter.py — removed from scipy years ago. The Modeler class doesn't need it, so importing directly around the broken __init__ works:

from MakeModel import Modeler   # not: from telfit import Modeler
After that LBLRTM runs took about 3 seconds per band over ~200–320 cm⁻¹, which was much faster than expected.
