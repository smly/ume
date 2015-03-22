# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import scipy.sparse as ss

from libc.stdlib cimport malloc, free
from libcpp.pair cimport pair
from libcpp.vector cimport vector


cdef extern from "wrapper/xgboost_wrapper.h":
    ctypedef unsigned long bst_ulong

    void *XGDMatrixCreateFromMat(
        const float *data,
        bst_ulong nrow,
        bst_ulong ncol,
        float missing)

    void *XGDMatrixCreateFromCSR(
        const bst_ulong *indptr,
        const unsigned *indices,
        const float *data,
        bst_ulong nindptr,
        bst_ulong nelem)

    void XGDMatrixSetFloatInfo(
        void *handle,
        const char *field,
        const float *array,
        bst_ulong length)

    void XGDMatrixFree(void *handle)

    void XGBoosterSetParam(void *handle, const char *k, const char *v)

    void *XGBoosterCreate(void *dmats[], bst_ulong length)

    void XGBoosterFree(void *handle)

    void XGBoosterUpdateOneIter(void *handle, int it, void *dtrain)

    const float *XGBoosterPredict(
        void *handle,
        void *dmat,
        int option_mask,
        unsigned ntree_limit,
        bst_ulong *length)


cdef class XGBoost:
    cdef void *booster
    cdef object params
    cdef void *dmats[2]  # 0: train, 1: test
    cdef object num_round

    def __cinit__(self, **cons_params):
        self.params = cons_params
        self.num_round = cons_params.get('num_round', 10)

    def __del__(self):
        XGBoosterFree(self.booster)
        XGDMatrixFree(self.dmats[0])
        XGDMatrixFree(self.dmats[1])

    def load_nd(self, X, y, idx, missing=0.0):
        cdef np.ndarray[float, ndim=1, mode='c'] data = np.array(
            X.reshape(X.size), dtype='float32')
        cdef void *dmat = XGDMatrixCreateFromMat(
            &data[0], X.shape[0], X.shape[1], missing)

        cdef np.ndarray[float, ndim=1, mode='c'] labels = np.array(
            y.reshape(y.size), dtype='float32')
        XGDMatrixSetFloatInfo(dmat, "label", &labels[0], y.size)
        self.dmats[idx] = dmat

    def load_ss(self, X, y, idx):
        assert len(X.indices) == len(X.data)
        nindptr = len(X.indptr)
        nelem = len(X.data)

        cdef bst_ulong *col_ptr = <bst_ulong*>malloc(len(X.indptr) * sizeof(bst_ulong))
        for i in range(len(X.indptr)): col_ptr[i] = X.indptr[i]
        cdef np.ndarray[unsigned, ndim=1, mode='c'] indices = np.array(
            X.indices, dtype='uint32')
        cdef np.ndarray[float, ndim=1, mode='c'] data = np.array(
            X.data, dtype='float32')

        cdef void *dmat = XGDMatrixCreateFromCSR(
            col_ptr, &indices[0], &data[0], nindptr, nelem)

        cdef np.ndarray[float, ndim=1, mode='c'] labels = np.array(y, dtype='float32')
        XGDMatrixSetFloatInfo(dmat, "label", &labels[0], y.size)
        self.dmats[idx] = dmat

    def set_param(self, k_str, v_str):
        k_byte_string = k_str.encode('utf-8')
        v_byte_string = v_str.encode('utf-8')
        cdef const char* param_k = k_byte_string
        cdef const char* param_v = v_byte_string
        XGBoosterSetParam(self.booster, param_k, param_v)

    def set_params(self):
        if isinstance(self.params, dict):
            for k, v in self.params.items():
                self.set_param(str(k), str(v))

    def setup_cache(self, X_tr, y_tr, X_ts):
        if isinstance(X_tr, np.ndarray):
            self.load_nd(X_tr, y_tr, 0)
        elif isinstance(X_tr, ss.csr_matrix):
            self.load_ss(X_tr, y_tr, 0)
        else:
            raise NotImplementedError("Unsupported data type")

        y_ts = np.zeros(X_ts.shape[0])
        if isinstance(X_ts, np.ndarray):
            self.load_nd(X_ts, y_ts, 1)
        elif isinstance(X_ts, ss.csr_matrix):
            self.load_ss(X_ts, y_ts, 1)
        else:
            raise NotImplementedError("Unsupported data type")

        self.booster = XGBoosterCreate(self.dmats, 2)
        self.set_param('seed', '0')
        self.set_params()

    def fit_predict(self, X_tr, y_tr, X_ts):
        self.setup_cache(X_tr, y_tr, X_ts)
        for i in range(self.num_round):
            XGBoosterUpdateOneIter(self.booster, i, self.dmats[0])

        ntree_limit = 0
        option_mask = 0x00

        cdef const float* preds_raw;
        cdef bst_ulong length;
        preds_raw = XGBoosterPredict(
            self.booster, self.dmats[1], option_mask,
            ntree_limit, &length)
        preds = np.array([preds_raw[i] for i in range(length)])

        num_class = self.params.get('num_class', 1)
        n_samples = length / num_class
        return preds.reshape((n_samples, num_class))
