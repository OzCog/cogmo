from libcpp.set cimport set
from libcpp.vector cimport vector
from atomspace cimport cHandle, cAtomSpace
from logger cimport cLogger


cdef extern from "opencog/ure/forwardchainer/ForwardChainer.h" namespace "opencog":
    cdef cppclass cForwardChainer "opencog::ForwardChainer":
        cForwardChainer(cAtomSpace& kb_as,
                        cAtomSpace& rb_as,
                        const cHandle& rbs,
                        const cHandle& source,
                        const cHandle& vardecl,
                        cAtomSpace* trace_as,
                        const vector[cHandle]& focus_set) except +

        void do_chain() except +
        cHandle get_results() const


cdef extern from "opencog/ure/backwardchainer/Fitness.h" namespace "opencog::BITNodeFitness":
    cdef cppclass BitNodeFitnessType:
        pass


cdef extern from "opencog/ure/backwardchainer/Fitness.h" namespace "opencog::AndBITNodeFitness":
    cdef cppclass AndBitFitnessType:
        pass

cdef extern from "opencog/ure/backwardchainer/Fitness.h" namespace "opencog::AndBITFitness::FitnessType":
    cdef AndBitFitnessType Uniform
    cdef AndBitFitnessType Trace


cdef extern from "opencog/ure/backwardchainer/Fitness.h" namespace "opencog::BITNodeFitness::FitnessType":
    cdef BitNodeFitnessType MaximizeConfidence


cdef extern from "opencog/ure/backwardchainer/Fitness.h" namespace "opencog":
    cdef cppclass ContentHash:
        pass

    cdef cppclass BITNodeFitness:
        BITNodeFitness(BitNodeFitnessType ft) except +

    cdef cppclass AndBITFitness:
        AndBITFitness(AndBitFitnessType ft,
                      const set[ContentHash]& tr) except +


cdef extern from "opencog/ure/backwardchainer/BackwardChainer.h" namespace "opencog":
    cdef cppclass cBackwardChainer "opencog::BackwardChainer":
        cBackwardChainer(cAtomSpace& _as,
                        const cHandle& rbs,
                        const cHandle& target,
                        const cHandle& vardecl,
                        cAtomSpace* trace_as,
                        cAtomSpace* control_as,
                        const cHandle& focus_set,
                        const BITNodeFitness& bitnode_fitness,
                        const AndBITFitness& andbit_fitness) except +

        cBackwardChainer(cAtomSpace& _as,
                        const cHandle& rbs,
                        const cHandle& target,
                        const cHandle& vardecl,
                        cAtomSpace* trace_as,
                        cAtomSpace* control_as,
                        const cHandle& focus_set) except +

        void do_chain() except +
        cHandle get_results() const


cdef extern from "opencog/ure/URELogger.h" namespace "opencog":
    cdef cLogger& ure_logger()
