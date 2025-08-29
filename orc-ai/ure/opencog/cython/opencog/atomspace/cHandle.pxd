# atomspace/cHandle.pxd
# Basic handle and atom type definitions

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

# Forward declarations
cdef extern from "opencog/atoms/base/Atom.h" namespace "opencog":
    cdef cppclass cAtom "opencog::Atom"

cdef extern from "opencog/atoms/base/Handle.h" namespace "opencog":
    ctypedef shared_ptr[cAtom] cAtomPtr "opencog::AtomPtr"
    
    cdef cppclass cHandle "opencog::Handle" (cAtomPtr):
        cHandle()
        cHandle(const cHandle&)
        
        cAtom* atom_ptr()
        
        bool operator==(cHandle h)
        bool operator!=(cHandle h)
        bool operator<(cHandle h)
        bool operator>(cHandle h)
        bool operator<=(cHandle h)
        bool operator>=(cHandle h)
    
    cdef cHandle UNDEFINED "opencog::Handle::UNDEFINED"