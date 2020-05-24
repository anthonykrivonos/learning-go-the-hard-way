package matrix

import (
	"fmt"
	"errors"
	rand "math/rand"
	"time"
)

type Matrix struct {
	shape []int
	content [][]float64
}


/* CONSTRUCTORS */

/**
 *	Constructs a new matrix copy from a matrix.
 */
func MatrixCopy(m *Matrix) *Matrix {
	copy := new(Matrix)
	copy.shape = m.shape
	copy.content = m.content
	return copy
}

/**
 *	Constructs a new matrix from a 2D slice.
 */
func MatrixFromArray(x [][]float64) *Matrix {
	m := new(Matrix)
	m.shape = []int{len(x), len(x[0])}
	m.content = x
	return m
}

/**
 *	Constructs a new matrix from a 2D slice.
 */
func MatrixFromDims(rows, columns int) *Matrix {
	m := new(Matrix)
	m.content = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		m.content[i] = make([]float64, columns)
	}
	m.shape = []int{rows, columns}
	return m
}

/**
 *	Constructs a matrix filled with ones.
 */
func MatrixOfOnes(rows, columns int) *Matrix {
	m := MatrixFromDims(rows, columns)
	oneFunc := func(x float64) float64 { return 1 }
	Vectorize(m, oneFunc)
	return m
}

/**
 *	Constructs a matrix filled with zeros.
 */
func MatrixOfZeros(rows, columns int) *Matrix {
	m := MatrixFromDims(rows, columns)
	zeroFunc := func(x float64) float64 { return 0 }
	Vectorize(m, zeroFunc)
	return m
}

/**
 *	Constructs an identity matrix of the given dimensions.
 */
func MatrixIdentity(rows, columns int) *Matrix {
	m := MatrixFromDims(rows, columns)
	identityFunc := func(x float64, i, j int) float64 {
		if i == j {
			return 1
		}
		return 0
	}
	VectorizeWithDims(m, identityFunc)
	return m
}

/**
 *	Constructs a matrix of normally distributed samples with the given dimensions.
 */
func MatrixStdNorm(rows, columns int) *Matrix {
	rand.Seed(int64(time.Now().Unix()))
	m := MatrixFromDims(rows, columns)
	normFunc := func(x float64) float64 { return rand.NormFloat64() }
	Vectorize(m, normFunc)
	return m
}


/* GETTERS AND SETTERS */

/**
 *	Sets the matrix to the value at the given coordinates i and j.
 */
func Set(m *Matrix, i, j int, value float64) {
	m.content[i][j] = value
}

/**
 *	Gets the matrix value at the given coordinates i and j.
 */
func Get(m *Matrix, i, j int) float64 {
	return m.content[i][j]
}


/* MATH FUNCTIONS */

/**
 *	Multiplies all values in the matrix by a scalar.
 */
func Add(lhs *Matrix, rhs *Matrix) (Matrix, error) {
	// Ensure the matrices are summable
	if lhs.shape[0] != rhs.shape[0] || lhs.shape[1] != rhs.shape[1] {
		return *lhs, errors.New("Incorrect shapes for adding: (" + string(lhs.shape[0]) + ", " + string(lhs.shape[1]) + ") and (" + string(rhs.shape[0]) + ", " + string(rhs.shape[1]) + ")")
	}

	// Create a new matrix with the same shape and add lhs plus rhs
	res := MatrixFromDims(lhs.shape[0], lhs.shape[1])
	add := func(x float64, i, j int) float64 { return Get(lhs, i, j) + Get(rhs, i, j) }
	VectorizeWithDims(res, add)

	return *res, nil
}

/**
 *	Adds a scalar to all values in the matrix.
 */
func ScalarAdd(m *Matrix, s float64) {
	addS := func(x float64) float64 { return x + s }
	Vectorize(m, addS)
}

/**
 *	Multiplies two matrices and returns the resulting matrix.
 */
func Multiply(lhs *Matrix, rhs *Matrix) (Matrix, error) {

	// Ensure the matrices are multiplicable
	if lhs.shape[1] != rhs.shape[0] {
		return *lhs, errors.New("Incorrect shapes " + string(lhs.shape[1]) + " and " + string(rhs.shape[0]))
	}

	// Create a new matrix from the outter dims of the given matrices
	res := MatrixFromDims(lhs.shape[0], rhs.shape[1])

	// Perform matrix multiplication
	for i := 0; i < lhs.shape[0]; i++ {
		for j := 0; j < rhs.shape[1]; j++ {
			for k := 0; k < rhs.shape[0]; k++ {
				new_val := Get(res, i, j) + (Get(lhs, i, k) * Get(rhs, k, j))
				Set(res, i, j, new_val)
			}
		}
	}

	return *res, nil
}

/**
 *	Multiplies all values in the matrix by a scalar.
 */
func ScalarMultiply(m *Matrix, s float64) {
	multS := func(x float64) float64 { return x * s }
	Vectorize(m, multS)
}


/* UTILITY FUNCTIONS */

/**
 *	Performs the given function on all values in the matrix.
 */
func Vectorize(m *Matrix, f func(float64) float64) {
	for i := 0; i < m.shape[0]; i++ {
		for j := 0; j < m.shape[1]; j++ {
			Set(m, i, j, f(Get(m, i, j)))
		}
	}
}

/**
 *	Performs the given function on all values in the matrix, with provided dims.
 */
func VectorizeWithDims(m *Matrix, f func(float64, int, int) float64) {
	for i := 0; i < m.shape[0]; i++ {
		for j := 0; j < m.shape[1]; j++ {
			Set(m, i, j, f(Get(m, i, j), i, j))
		}
	}
}

/**
 *	Prints a matrix, row by row.
 */
func PrintMatrix(m *Matrix) {

	for i := 0; i < m.shape[0]; i++ {
		for j := 0; j < m.shape[1]; j++ {
			m_ij := Get(m, i, j)
			if m_ij >= 0 {
				fmt.Printf(" ")
			}
			fmt.Printf("%.5f", m_ij)
			if j != m.shape[1] - 1 {
				fmt.Printf(" ")
			}
		}
		fmt.Printf("\n")
	}
}