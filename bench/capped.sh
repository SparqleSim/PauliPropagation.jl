#!/usr/bin/env bash
# Run a command under a kernel-enforced memory cap. Launch every benchmark through this.
#
# An in-process check cannot help: the dangerous allocation is LLVM recompiling the pipeline for each
# new integer width, which no Julia-side poll sees coming or can abort. MemorySwapMax=0 makes an
# overrun fail fast instead of thrashing swap.
#
# usage: bench/capped.sh [limit] <command...>          (limit defaults to 4G, e.g. 2G, 512M)
set -uo pipefail

LIMIT=4G
case "${1:-}" in
  [0-9]*[MG]) LIMIT="$1"; shift ;;
esac

if [ $# -eq 0 ]; then
  echo "usage: bench/capped.sh [limit] <command...>" >&2
  exit 2
fi

systemd-run --user --scope -p MemoryMax="$LIMIT" -p MemorySwapMax=0 --quiet -- "$@"
status=$?

if [ $status -eq 137 ]; then
  echo "" >&2
  echo "KILLED: exceeded the $LIMIT memory cap. The machine is fine; the run is not." >&2
  echo "Shrink the widest Pauli string type before retrying -- the compile, not the sum, is the cost." >&2
fi

exit $status
