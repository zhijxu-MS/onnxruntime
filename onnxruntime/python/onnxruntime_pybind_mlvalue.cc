// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_mlvalue.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/graph/graph.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"

using namespace std;
namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime::logging;

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type) {
  static std::map<MLDataType, int> type_map{
      {DataTypeImpl::GetType<bool>(), NPY_BOOL},
      {DataTypeImpl::GetType<float>(), NPY_FLOAT},
      {DataTypeImpl::GetType<double>(), NPY_DOUBLE},
      {DataTypeImpl::GetType<int32_t>(), NPY_INT},
      {DataTypeImpl::GetType<int8_t>(), NPY_INT8},
      {DataTypeImpl::GetType<uint8_t>(), NPY_UINT8},
      {DataTypeImpl::GetType<int16_t>(), NPY_INT16},
      {DataTypeImpl::GetType<uint16_t>(), NPY_UINT16},
      {DataTypeImpl::GetType<int64_t>(), NPY_LONGLONG},
      {DataTypeImpl::GetType<uint64_t>(), NPY_ULONGLONG},
      {DataTypeImpl::GetType<std::string>(), NPY_OBJECT},
  };

  const auto it = type_map.find(tensor_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for Tensor Type.");
  } else {
    return it->second;
  }
}

const DataTypeImpl* NumpyToOnnxRuntimeTensorType(int numpy_type) {
  static std::map<int, MLDataType> type_map{
      {NPY_BOOL, DataTypeImpl::GetType<bool>()},
      {NPY_FLOAT, DataTypeImpl::GetType<float>()},
      {NPY_DOUBLE, DataTypeImpl::GetType<double>()},
      {NPY_INT, DataTypeImpl::GetType<int32_t>()},
      {NPY_INT8, DataTypeImpl::GetType<int8_t>()},
      {NPY_UINT8, DataTypeImpl::GetType<uint8_t>()},
      {NPY_INT16, DataTypeImpl::GetType<int16_t>()},
      {NPY_UINT16, DataTypeImpl::GetType<uint16_t>()},
      {NPY_LONG,
       sizeof(long) == sizeof(int) ? DataTypeImpl::GetType<int32_t>()
                                   : DataTypeImpl::GetType<int64_t>()},
      {NPY_LONGLONG, DataTypeImpl::GetType<int64_t>()},
      {NPY_ULONGLONG, DataTypeImpl::GetType<uint64_t>()},
      {NPY_UNICODE, DataTypeImpl::GetType<std::string>()},
      {NPY_STRING, DataTypeImpl::GetType<std::string>()},
      {NPY_OBJECT, DataTypeImpl::GetType<std::string>()},
      {NPY_VOID, DataTypeImpl::GetType<std::string>()}};

  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error("Numpy_type " + std::to_string(numpy_type) +
                             " can't be converted to MLDataType.");
  } else {
    return it->second;
  }
}

bool PyObjectCheck_Array(PyObject* o) {
  return PyObject_HasAttrString(o, "__array_finalize__");
}

void CreateTensorMLValue(AllocatorPtr alloc, const std::string& name_input, PyArrayObject* pyObject, MLValue* p_mlvalue) {
  PyArrayObject* darray = PyArray_GETCONTIGUOUS(pyObject);
  if (darray == NULL) {
    throw std::runtime_error(std::string("The object must be a contiguous array for input '") + name_input + std::string("'."));
  }
  bool dref = false;
  try {
    const int npy_type = PyArray_TYPE(darray);

    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(darray);
    npy_intp* npy_dims = PyArray_DIMS(darray);
    std::vector<int64_t> dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      dims[i] = npy_dims[i];
    }

    TensorShape shape(dims);
    auto element_type = NumpyToOnnxRuntimeTensorType(npy_type);
    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type, shape, alloc);
    if (npy_type == NPY_UNICODE) {
      // Copy string data which needs to be done after Tensor is allocated.
      // Strings are Python strings or numpy.unicode string.
      std::string* dst = p_tensor->MutableData<std::string>();
      auto item_size = PyArray_ITEMSIZE(darray);
      auto num_chars = item_size / PyUnicode_4BYTE_KIND;
      char* src = static_cast<char*>(PyArray_DATA(darray));
      const char* str;
      Py_ssize_t size;
      PyObject* pStr;
      for (int i = 0; i < shape.Size(); i++, src += item_size) {
        // Python unicode strings are assumed to be USC-4. Strings are stored as UTF-8.
        pStr = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, src, num_chars);
        str = PyUnicode_AsUTF8AndSize(pStr, &size);
        if (str == NULL) {
          dst[i] = "";
        } else {
          // Size is equal to the longest string size, numpy stores
          // strings in a single array. Those code assumes a string ends with a final 0.
          dst[i] = str;
        }
        Py_XDECREF(pStr);
      }
    } else if (npy_type == NPY_STRING || npy_type == NPY_VOID) {
      // Copy string data which needs to be done after Tensor is allocated.
      // Strings are given as bytes (encoded strings).
      // NPY_VOID does not trim final 0.
      // NPY_STRING assumes bytes string ends with a final 0.
      std::string* dst = p_tensor->MutableData<std::string>();
      auto item_size = PyArray_ITEMSIZE(darray);
      char* src = static_cast<char*>(PyArray_DATA(darray));
      for (int i = 0; i < shape.Size(); i++, src += item_size) {
        if (npy_type == NPY_STRING) {
          dst[i] = src;
        } else {
          dst[i].resize(item_size);
          memcpy((void*)dst[i].c_str(), src, item_size);
        }
      }
    } else if (npy_type == NPY_OBJECT) {
      // Converts object into string.
      std::string* dst = p_tensor->MutableData<std::string>();
      auto item_size = PyArray_ITEMSIZE(darray);
      char* src = static_cast<char*>(PyArray_DATA(darray));
      PyObject *item, *pStr;
      for (int i = 0; i < shape.Size(); ++i, src += item_size) {
        // Python unicode strings are assumed to be USC-4. Strings are stored as UTF-8.
        item = PyArray_GETITEM(darray, src);
        pStr = PyObject_Str(item);
        dst[i] = py::reinterpret_borrow<py::str>(pStr);
        Py_XDECREF(pStr);
      }
    } else {
      void* buffer = p_tensor->MutableDataRaw();
      size_t len;
      if (!IAllocator::CalcMemSizeForArray(element_type->Size(), shape.Size(), &len)) {
        throw std::runtime_error("length overflow");
      }
      memcpy(buffer, static_cast<void*>(PyArray_DATA(darray)), len);
    }

    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  } catch (...) {
    if (!dref) {
      Py_XDECREF(darray);
      dref = true;
    }

    // allocator should be able to gc the memory created by it.
    // ...

    throw;
  }

  if (!dref) {
    Py_XDECREF(darray);
  }
}

std::string _get_type_name(int64_t&) {
  return std::string("int64_t");
}

std::string _get_type_name(float&) {
  return std::string("float");
}

std::string _get_type_name(std::string&) {
  return std::string("string");
}

template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
void CreateMapMLValue_LoopIntoMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                  PyObject* item, std::map<KeyType, ValueType>& current,
                                  KeyGetterType keyGetter, ValueGetterType valueGetter) {
  KeyType ckey;
  ValueType cvalue;
  do {
    if (!keyGetter(key, ckey)) {
      PyObject* pType = PyObject_Type(key);
      auto pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);
      Py_XDECREF(item);
      throw std::runtime_error(std::string("Unexpected key type  ") + sType +
                               std::string(", it cannot be linked to C type ") +
                               _get_type_name(ckey) + std::string(" for input '") +
                               name_input + std::string("'."));
    }

    if (!valueGetter(value, cvalue)) {
      PyObject* pType = PyObject_Type(value);
      auto pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);
      Py_XDECREF(item);
      throw std::runtime_error(std::string("Unexpected value type  ") + sType +
                               std::string(", it cannot be linked to C type ") +
                               _get_type_name(ckey) + std::string(" for input '") +
                               name_input + std::string("'."));
    }
    current[ckey] = cvalue;
  } while (PyDict_Next(item, &pos, &key, &value));
}

template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
void CreateMapMLValue_Map(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                          PyObject* item,
                          AllocatorPtr /*alloc*/, MLValue* p_mlvalue,
                          KeyGetterType keyGetter, ValueGetterType valueGetter) {
  std::unique_ptr<std::map<KeyType, ValueType>> dst;
  dst = std::make_unique<std::map<KeyType, ValueType>>();
  CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, *dst, keyGetter, valueGetter);
  p_mlvalue->Init(dst.release(), DataTypeImpl::GetType<std::map<KeyType, ValueType>>(),
                  DataTypeImpl::GetType<std::map<KeyType, ValueType>>()->GetDeleteFunc());
}

template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
void CreateMapMLValue_VectorMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                PyObject* iterator, PyObject* item,
                                AllocatorPtr /*alloc*/, MLValue* p_mlvalue,
                                KeyGetterType keyGetter, ValueGetterType valueGetter) {
  std::unique_ptr<std::vector<std::map<KeyType, ValueType>>> dstVector;
  dstVector = std::make_unique<std::vector<std::map<KeyType, ValueType>>>();
  int index = 0;
  do {
    dstVector->push_back(std::map<KeyType, ValueType>());
    CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, (*dstVector)[index], keyGetter, valueGetter);
    Py_DECREF(item);
    ++index;
    item = iterator == NULL ? NULL : PyIter_Next(iterator);
  } while (item != NULL);
  p_mlvalue->Init(dstVector.release(), DataTypeImpl::GetType<std::vector<std::map<KeyType, ValueType>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<KeyType, ValueType>>>()->GetDeleteFunc());
}

void CreateMapMLValue_AgnosticMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                  PyObject* iterator, PyObject* item,
                                  AllocatorPtr alloc, MLValue* p_mlvalue) {
  // If iterator is NULL, it returns a single Map,
  // if is not NULL, it returns a VectorMap.
  auto int64Getter = [](PyObject* obj, int64_t& value) -> bool {
    value = PyLong_AsLong(obj);
    return !PyErr_Occurred();
  };

  auto floatGetter = [](PyObject* obj, float& value) -> bool {
    if (PyFloat_Check(obj)) {
      value = (float)PyFloat_AS_DOUBLE(obj);
      return true;
    } else if (PyNumber_Check(obj)) {
      value = (float)PyFloat_AsDouble(obj);
      return true;
    } else {
      return false;
    }
  };

  auto stringGetter = [](PyObject* obj, std::string& value) -> bool {
    PyObject* pStr = PyObject_Str(obj);
    if (pStr == NULL) {
      return false;
    }
    value = py::reinterpret_borrow<py::str>(pStr);
    Py_DECREF(pStr);
    return true;
  };

  if (iterator == NULL) {
    if (PyLong_Check(key)) {
      // Regular Python.
      CreateMapMLValue_Map<int64_t, float>(pos, key, name_input, value, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyNumber_Check(key)) {
      // For numpy type.
      CreateMapMLValue_Map<int64_t, float>(pos, key, name_input, value, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyUnicode_Check(key)) {
      CreateMapMLValue_Map<std::string, float>(pos, key, name_input, value, item, alloc, p_mlvalue, stringGetter, floatGetter);
    } else {
      PyObject* pType = PyObject_Type(key);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Key type must be int or string (not ") + sType +
                               std::string(") for input '") + name_input + std::string("'."));
    }
  } else {
    if (PyLong_Check(key)) {
      CreateMapMLValue_VectorMap<int64_t, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyNumber_Check(key)) {
      // For numpy type.
      CreateMapMLValue_VectorMap<int64_t, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyUnicode_Check(key)) {
      CreateMapMLValue_VectorMap<std::string, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, stringGetter, floatGetter);
    } else {
      PyObject* pType = PyObject_Type(value);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Key type must be int or string (not ") + sType +
                               std::string(") for input '") + name_input + std::string("'."));
    }
  }
}

void CreateMapMLValue_AgnosticVectorMap(PyObject* iterator, PyObject* item, AllocatorPtr alloc, const std::string& name_input, MLValue* p_mlvalue) {
  // CreateMapMLValue is called by CreateGenericTerableMLValue
  // or CreateGenericMLValue which ensures
  // item is a dictionary, no need to check type again.
  // This functions starts to iterate on the first
  // element of the dictionary and calls CreateMapMLValue_AgnosticMap
  // which determines the container type. This type
  // is based on the first pair of the dictionary
  // and all the function assumes the key and value type remain the same
  // for all pairs in the dictionary.

  // If iterator is NULL, it returns a single Map,
  // if is not NULL, it returns a VectorMap.

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  if (PyDict_Next(item, &pos, &key, &value)) {
    CreateMapMLValue_AgnosticMap(pos, key, name_input, value, iterator, item, alloc, p_mlvalue);
  } else {
    throw std::runtime_error("Size of dictionary is empty, unable to run the prediction.");
  }
}

void CreateGenericIterableMLValue(PyObject* iterator, AllocatorPtr alloc, const std::string& name_input, MLValue* p_mlvalue) {
  PyObject* item;
  MLValue ml_value;
  item = PyIter_Next(iterator);
  if (item == NULL) {
    throw std::runtime_error("Input '" + name_input + "' must not be empty.");
  }
  if (PyObjectCheck_Array(item)) {
    PyObject* pType = PyObject_Type(item);
    PyObject* pStr = PyObject_Str(pType);
    py::str spyType = py::reinterpret_borrow<py::str>(pStr);
    std::string sType = spyType;
    Py_XDECREF(pType);
    Py_XDECREF(pStr);
    throw std::runtime_error("Iterable of " + sType + " should be given as array for input '" +
                             name_input + std::string("'."));
  } else {
    // We expect a dictionary.
    if (!PyDict_Check(item)) {
      throw std::runtime_error("Input must be a list of dictionaries or a single numpy array for input '" +
                               name_input + std::string("'."));
    }
    CreateMapMLValue_AgnosticVectorMap(iterator, item, alloc, name_input, p_mlvalue);
  }
}

void CreateGenericMLValue(AllocatorPtr alloc, const std::string& name_input, py::object& value, MLValue* p_mlvalue) {
  if (PyObjectCheck_Array(value.ptr())) {
    // The most frequent case: input comes as an array.
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(value.ptr());
    CreateTensorMLValue(alloc, name_input, arr, p_mlvalue);
  } else if (PyDict_Check(value.ptr())) {
    CreateMapMLValue_AgnosticVectorMap((PyObject*)NULL, value.ptr(), alloc, name_input, p_mlvalue);
  } else {
    auto iterator = PyObject_GetIter(value.ptr());
    if (iterator == NULL) {
      // The pype cannot be handled.
      PyObject* pType = PyObject_Type(value.ptr());
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Unable to handle object of type ") + sType);
    }
    // We assume the object is iterable.
    // iterator should not be NULL due to previous test.
    try {
      CreateGenericIterableMLValue(iterator, alloc, name_input, p_mlvalue);
    } catch (const std::runtime_error&) {
      Py_DECREF(iterator);
      throw;
    }
    Py_DECREF(iterator);
  }
}

}  // namespace python
}  // namespace onnxruntime
