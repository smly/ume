# -*- coding: utf-8 -*-


cdef extern from "libjsonnet.h":
    cdef cppclass JsonnetVm:
        JsonnetVm()
    JsonnetVm* jsonnet_make()
    void jsonnet_max_stack(JsonnetVm *vm, unsigned v)
    void jsonnet_gc_min_objects(JsonnetVm* vm, unsigned v)
    void jsonnet_gc_growth_trigger(JsonnetVm* vm, double v)
    void jsonnet_debug_ast(JsonnetVm* vm, int v)
    void jsonnet_max_trace(JsonnetVm* vm, unsigned v)
    void jsonnet_destroy(JsonnetVm* vm)
    #void jsonnet_cleanup_string(JsonnetVm* vm, const char*)
    const char* jsonnet_evaluate_file(JsonnetVm* vm, const char* filename, int* error)
    const char* jsonnet_evaluate_snippet(JsonnetVm* vm, const char* filename, const char* snippet, int* error)


def load(filename):
    cdef JsonnetVm* vm
    cdef const char* src
    cdef const char* out
    cdef unsigned max_stack = 500, gc_min_objects = 1000, max_trace = 20
    cdef double gc_growth_trigger = 2
    cdef int debug_ast = 0, error

    vm = jsonnet_make()
    jsonnet_max_stack(vm, max_stack)
    jsonnet_gc_min_objects(vm, gc_min_objects)
    jsonnet_max_trace(vm, max_trace)
    jsonnet_gc_growth_trigger(vm, gc_growth_trigger)
    jsonnet_debug_ast(vm, debug_ast)

    cdef bytes py_bytes = filename.encode()
    cdef const char* c_string = py_bytes

    out = jsonnet_evaluate_file(vm, c_string, &error)
    cdef bytes python_byte_s = out
    #jsonnet_cleanup_string(vm, out)
    jsonnet_destroy(vm)

    return python_byte_s


def loads(code):
    cdef JsonnetVm* vm
    cdef const char* filename = "rawcode"
    cdef const char* src
    cdef const char* out
    cdef unsigned max_stack = 500, gc_min_objects = 1000, max_trace = 20
    cdef double gc_growth_trigger = 2
    cdef int debug_ast = 0, error

    vm = jsonnet_make()
    jsonnet_max_stack(vm, max_stack)
    jsonnet_gc_min_objects(vm, gc_min_objects)
    jsonnet_max_trace(vm, max_trace)
    jsonnet_gc_growth_trigger(vm, gc_growth_trigger)
    jsonnet_debug_ast(vm, debug_ast)

    cdef bytes py_bytes = code.encode()
    cdef const char* c_string = py_bytes

    out = jsonnet_evaluate_snippet(vm, filename, c_string, &error)
    cdef bytes python_byte_s = out
    #jsonnet_cleanup_string(vm, out)
    jsonnet_destroy(vm)

    return python_byte_s
