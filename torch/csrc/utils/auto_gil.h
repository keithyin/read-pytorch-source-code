#pragma once

// RAII structs to acquire and release Python's global interpreter lock (GIL)

#include <Python.h>

// Acquires the GIL on construction
// 每当从 C 访问 Python 时， 需要保证 对 GIL 做合适的获取和释放动作！！
/*
PyGILState_Ensure(); 
PyGILState_Release(); 使 python 的解释器恢复到之前的状态
*/
struct AutoGIL {
  AutoGIL() : gstate(PyGILState_Ensure()) {
  }
  ~AutoGIL() {
    PyGILState_Release(gstate);
  }

  PyGILState_STATE gstate;
};

// Releases the GIL on construction

/*

*/
struct AutoNoGIL {
  AutoNoGIL() : save(PyEval_SaveThread()) {
  }
  ~AutoNoGIL() {
    PyEval_RestoreThread(save);
  }

  PyThreadState* save;
};
