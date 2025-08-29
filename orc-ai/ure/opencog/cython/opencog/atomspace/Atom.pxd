# atomspace/Atom.pxd
# Atom type definitions with createAtom function

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from .cHandle cimport cHandle, cAtom
from .AtomSpace cimport Value, cValue, cValuePtr, tv_ptr, Type

# ContentHash
ctypedef size_t ContentHash

# Extended Atom definition
cdef extern from "opencog/atoms/base/Atom.h" namespace "opencog":
    cdef cppclass cAtom "opencog::Atom" (cValue):
        cAtom()
        
        tv_ptr getTruthValue()
        void setTruthValue(tv_ptr tvp)
        void setValue(const cHandle& key, const cValuePtr& value)
        cValuePtr getValue(const cHandle& key) const
        cpp_set[cHandle] getKeys()
        
        string to_string()
        string to_short_string()
        string id_to_string()
        
        # Conditionally-valid methods. Not defined for all atoms.
        string get_name()
        vector[cHandle] getOutgoingSet()
        ContentHash get_hash()
        
        bool operator==(cAtom&)
        bool operator<(cAtom&)

# Forward declaration for AtomSpace
cdef extern from "opencog/atomspace/AtomSpace.h" namespace "opencog":
    cdef cppclass cAtomSpace "opencog::AtomSpace"

# Extend cAtom with AtomSpace reference
cdef extern from "opencog/atoms/base/Atom.h" namespace "opencog":
    cdef cppclass cAtom "opencog::Atom":
        cAtomSpace* getAtomSpace()

# Atom wrapper class with createAtom function
cdef class Atom(Value):
    cdef object _atom_type
    cdef object _name  
    cdef object _outgoing
    cdef cHandle get_c_handle(Atom self)
    
    @staticmethod
    cdef Atom createAtom(cHandle& handle)