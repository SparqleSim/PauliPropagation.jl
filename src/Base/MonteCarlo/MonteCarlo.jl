###
##
# Monte Carlo propagation, built on top of the Propagation module: stochastic path sampling
# (mcsample) and periodic resampling of an oversized ensemble (mcpropagate, via resample).
##
###

include("./resample.jl")
include("./mcsample.jl")
include("./mcpropagate.jl")
