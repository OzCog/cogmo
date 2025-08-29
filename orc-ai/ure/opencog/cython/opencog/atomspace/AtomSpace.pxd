# atomspace/AtomSpace.pxd  
# AtomSpace type definitions

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from .cHandle cimport cHandle

# Basic types
ctypedef short Type

# Value type for truth values
cdef extern from "opencog/atoms/value/Value.h" namespace "opencog":
    cdef cppclass cValue "opencog::Value":
        Type get_type()
        bool is_atom()
        bool is_node()
        bool is_link()
        
        string to_string()
        string to_short_string()
        bool operator==(const cValue&)
        bool operator!=(const cValue&)
    
    ctypedef shared_ptr[cValue] cValuePtr "opencog::ValuePtr"

# TruthValue types  
ctypedef double count_t
ctypedef double confidence_t
ctypedef double strength_t

cdef extern from "opencog/atoms/truthvalue/TruthValue.h" namespace "opencog":
    ctypedef shared_ptr[const cTruthValue] tv_ptr "opencog::TruthValuePtr"
    cdef cppclass cTruthValue "const opencog::TruthValue"(cValue):
        strength_t get_mean()
        confidence_t get_confidence()
        count_t get_count()

# AtomSpace
cdef extern from "opencog/atomspace/AtomSpace.h" namespace "opencog":
    cdef cppclass cAtomSpace "opencog::AtomSpace":
        cHandle add_atom(cHandle handle) except +
        
        cHandle xadd_node(Type t, string s) except +
        cHandle add_node(Type t, string s, tv_ptr tvn) except +
        
        cHandle xadd_link(Type t, vector[cHandle]) except +
        cHandle add_link(Type t, vector[cHandle], tv_ptr tvn) except +
        
        cHandle get_handle(Type t, string s)
        cHandle get_handle(Type t, vector[cHandle])
        
        cHandle set_value(cHandle h, cHandle key, cValuePtr value)
        cHandle set_truthvalue(cHandle h, tv_ptr tvn)
        cHandle get_atom(cHandle & h)
        bool is_valid_handle(cHandle h)
        int get_size()
        string get_name()
        
        # ==== query methods ====
        # get by type
        void get_handles_by_type(vector[cHandle], Type t, bool subclass)
        
        void clear()
        bool extract_atom(cHandle h, bool recursive)

    cdef cValuePtr createAtomSpace(cAtomSpace *parent)

# PtrHolder for managing shared pointers
cdef class PtrHolder:
    cdef shared_ptr[void] shared_ptr

# Forward declarations for Python wrapper classes
cdef class Value:
    cdef PtrHolder ptr_holder
    cdef cValuePtr get_c_value_ptr(self)

cdef class AtomSpace(Value):
    cdef cValuePtr asp
    cdef cAtomSpace *atomspace
    cdef object parent_atomspace