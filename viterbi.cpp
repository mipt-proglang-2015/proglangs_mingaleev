#include <vector>
#include <iostream>
#include <limits>

extern "C" {
#include <Python.h>
}

namespace viterbi {

    static std::vector<int> FindHiddenStates(const std::vector<int> & states, 
                                      const std::vector<double> & startProbs, 
                                      const std::vector<int> & observation,
                                      const std::vector<std::vector<double> > & transitionProbs,
                                      const std::vector<std::vector<double> > & emissionProbs) {
        int statesCount = states.size();
        int obsCount = observation.size();

        std::vector<std::vector<double> > TState = std::vector<std::vector<double> >(statesCount, 
                                                                                   std::vector<double>(obsCount));
        std::vector<std::vector<int> >    TIndex = std::vector<std::vector<int> >(statesCount, 
                                                                                std::vector<int>(obsCount));

        for (int index = 0; index < statesCount; ++index) {
            TState[index][0] = startProbs[index] * emissionProbs[index][observation[index]];
            TIndex[index][0] = 0;
        }

        for (int obsIndex = 1; obsIndex < obsCount; ++obsIndex) {
            for (int stateIndex = 0; stateIndex < statesCount; ++stateIndex) {
                TState[stateIndex][obsIndex] = TState[0][obsIndex - 1] * transitionProbs[0][stateIndex] * emissionProbs[stateIndex][observation[obsIndex]];
                TIndex[stateIndex][obsIndex] = 0;

                for (int k = 0; k < statesCount; ++k) {
                    if (TState[stateIndex][obsIndex] < 
                                         TState[k][obsIndex - 1] 
                                         * transitionProbs[k][stateIndex] 
                                         * emissionProbs[stateIndex][observation[obsIndex]]) {
                        TIndex[stateIndex][obsIndex] = k;
                        TState[stateIndex][obsIndex] = TState[k][obsIndex - 1] * transitionProbs[k][stateIndex] * emissionProbs[stateIndex][observation[obsIndex]];
                    }
                }
            }
        }

        std::vector<int> result(obsCount);
        result[obsCount - 1] = 1;

        for (int index = 0; index < statesCount; ++index) {
            if (TState[index][obsCount - 1] > TState[result[obsCount - 1]][obsCount - 1]) {
                result[obsCount - 1] = index;
            }
        }
        
        for (int index = obsCount - 1; index > 0; --index) {
            result[index - 1] = TIndex[result[index]][index];
        }
        
        return result;
    }
}

static std::vector<double> ConvectPyObjectToDoubleVector(PyObject * py_list) {
	std::vector<double> result(PyObject_Length(py_list));	

	for (int index = 0; index < result.size(); ++index) {
		PyObject* py_elem = PyList_GetItem(py_list, index);
        result[index] = PyFloat_AsDouble(py_elem);
	}

	return result;
}

static std::vector<int> ConvectPyObjectToIntVector(PyObject * py_list) {
    std::vector<int> result(PyObject_Length(py_list));

    for (int index = 0; index < result.size(); ++index) {
        PyObject* py_elem = PyList_GetItem(py_list, index);
        result[index] = PyLong_AsLong(py_elem);
    }

    return result;
}

static std::vector<std::vector<double> > ConvertPyObjectToDoubleMatrix(PyObject* py_matrix) {
    size_t size = PyObject_Length(py_matrix);    
	std::vector<std::vector<double> > result(size, std::vector<double>(size));
	
	for (int rowIndex = 0; rowIndex < size; ++rowIndex) {
		PyObject* py_row = PyList_GetItem(py_matrix, rowIndex);

		for (int colIndex = 0; colIndex < size; ++colIndex) {
			PyObject* py_elem = PyList_GetItem(py_row, colIndex);
            result[rowIndex][colIndex] = PyFloat_AsDouble(py_elem);
		}
	}

	return result;
}

static PyObject* ConvertVectorToPyObject(const std::vector<int>& cpp_elements) {
	PyObject* result = PyList_New(cpp_elements.size());

	for (int index = 0; index < cpp_elements.size(); ++index) {				
		PyObject* pyElem = Py_BuildValue("i", cpp_elements[index]);
		PyList_SetItem(result, index, pyElem);
	}

	return result;
}

static PyObject * FindHiddenStates(PyObject * module, PyObject * args)
{
	PyObject* py_states = PyTuple_GetItem(args, 0);
	PyObject* py_start_probs = PyTuple_GetItem(args, 1);
    PyObject* py_observations = PyTuple_GetItem(args, 2);
    PyObject* py_transitions_probs = PyTuple_GetItem(args, 3);
    PyObject* py_emission_probs = PyTuple_GetItem(args, 4);

	std::vector<int> states = ConvectPyObjectToIntVector(py_states);
	std::vector<double> start_probs = ConvectPyObjectToDoubleVector(py_start_probs);
    std::vector<int> observations = ConvectPyObjectToIntVector(py_observations);
    std::vector<std::vector<double> > transition_probs = ConvertPyObjectToDoubleMatrix(py_transitions_probs);    
    std::vector<std::vector<double> > emission_probs = ConvertPyObjectToDoubleMatrix(py_emission_probs);

	std::vector<int> result = viterbi::FindHiddenStates(states, start_probs, observations, transition_probs, emission_probs);
    
	PyObject * pyResult = ConvertVectorToPyObject(result);
	return pyResult;
}

PyMODINIT_FUNC PyInit_viterbi() {
	static PyMethodDef ModuleMethods[] = {
		{ "FindHiddenStates", FindHiddenStates, METH_VARARGS, "Realization of Viterbi algorithm" },
		{ NULL, NULL, 0, NULL }
	};

	static PyModuleDef ModuleDef = {
		PyModuleDef_HEAD_INIT,
		"viterbi",
		"Viterbi algorithm",
		-1, ModuleMethods,
		NULL, NULL, NULL, NULL
	};

	PyObject * module = PyModule_Create(&ModuleDef);
	return module;
}
