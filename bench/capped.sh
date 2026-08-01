#!/usr/bin/env bash
# Run a command under a kernel-enforced memory cap, so an overrun kills the command instead of the
# machine. Every benchmark in this directory should be launched through this.
#
# An in-process check is not enough. The dangerous allocation when sweeping qubit counts is not the
# Pauli sum, it is the compiler: each new qubit count makes `getinttype` define a fresh primitive
# integer type and the whole propagation pipeline is recompiled for it, and expanding operations on an
# integer hundreds of machine words wide costs LLVM superlinear memory. That is a malloc inside LLVM,
# invisible to `Sys.maxrss()` polling and impossible to abort partway. Only a cgroup limit stops it.
#
# `MemorySwapMax=0` matters as much as the limit itself: without it an overrun thrashes swap and takes
# the machine down slowly rather than failing fast.
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
