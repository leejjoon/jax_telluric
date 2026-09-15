#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reference_root="$project_root/data/lblrtm"
toolchain="$project_root/.conda-lblrtm"

mkdir -p "$reference_root"

if [[ ! -d "$reference_root/LBLRTM/.git" ]]; then
  git clone --depth 1 --branch v12.17 --recurse-submodules --shallow-submodules \
    https://github.com/AER-RC/LBLRTM.git "$reference_root/LBLRTM"
fi
if [[ ! -d "$reference_root/LNFL/.git" ]]; then
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/AER-RC/LNFL.git "$reference_root/LNFL"
fi
if [[ ! -d "$reference_root/AER_Line_File/.git" ]]; then
  git clone --depth 1 https://github.com/AER-RC/AER_Line_File.git \
    "$reference_root/AER_Line_File"
fi

if [[ ! -x "$toolchain/bin/x86_64-conda-linux-gnu-gfortran" ]]; then
  conda create -y -p "$toolchain" -c conda-forge \
    gfortran_linux-64 gcc_linux-64 netcdf-fortran make
fi
ln -sf x86_64-conda-linux-gnu-gfortran "$toolchain/bin/gfortran"

addlibs="$reference_root/LBLRTM/build/addlibs.inc"
sed -i "s|^NCL = .*|NCL = $toolchain/lib|" "$addlibs"
sed -i "s|^NCI = .*|NCI = $toolchain/include|" "$addlibs"

PATH="$toolchain/bin:$PATH" make -C "$reference_root/LBLRTM/build" \
  -f make_lblrtm linuxGNUsgl

lnfl_makefile="$reference_root/LNFL/build/makefile.common"
sed -i 's/FCFLAG="-Wall -frecord-marker=4"/FCFLAG="-Wall -frecord-marker=4 -std=legacy -fallow-argument-mismatch"/' \
  "$lnfl_makefile"
PATH="$toolchain/bin:$PATH" make -C "$reference_root/LNFL/build" \
  -f make_lnfl linuxGNUsgl

archive="$reference_root/AER_Line_File/aer_v_3.9.tgz"
archive_size=405878572
current_size=0
if [[ -f "$archive" ]]; then
  current_size=$(stat -c '%s' "$archive")
fi
if [[ "$current_size" -ne "$archive_size" ]]; then
  curl -L --fail --continue-at - --output "$archive" \
    'https://zenodo.org/records/18881607/files/aer_v_3.9.tgz?download=1'
fi
tar -tzf "$archive" >/dev/null
if [[ ! -d "$reference_root/AER_Line_File/aer_v_3.9" ]]; then
  tar -xzf "$archive" -C "$reference_root/AER_Line_File"
fi

lnfl_run="$reference_root/run_lnfl_igrins"
mkdir -p "$lnfl_run"
cat > "$lnfl_run/TAPE5" <<'EOF'
LBLRTM 12.17 IGRINS H and K line file
  4000.000  6750.000
11111110000000000000000000000000000000000000000
EOF
ln -sf "$reference_root/LNFL/lnfl_v3.2_linux_gnu_sgl" "$lnfl_run/lnfl"
ln -sf "$reference_root/AER_Line_File/aer_v_3.9/line_file/aer_v_3.9" "$lnfl_run/TAPE1"
if [[ ! -s "$lnfl_run/TAPE3" ]]; then
  (cd "$lnfl_run" && ./lnfl)
fi

printf 'LBLRTM: %s\n' "$reference_root/LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
printf 'LNFL:    %s\n' "$reference_root/LNFL/lnfl_v3.2_linux_gnu_sgl"
printf 'Lines:   %s\n' "$reference_root/AER_Line_File/aer_v_3.9"
printf 'TAPE3:   %s\n' "$lnfl_run/TAPE3"
