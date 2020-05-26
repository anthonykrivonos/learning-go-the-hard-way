package matrix

import (
	"fmt"
	"errors"
	math "math"
	rand "math/rand"
	"time"
)

type Matrix struct {
	Shape []int
	content [][]float64
}


/* CONSTRUCTORS */

/**
 *	Constructs a new matrix copy from a matrix.
 */
func MatrixCopy(m *Matrix) *Matrix {
	copy := MatrixFromDims(m.Shape[0], m.Shape[1])
	copy.Shape = []int{m.Shape[0], m.Shape[1]}
	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			Set(copy, i, j, Get(m, i, j))
		}
	}
	return copy
}

/**
 *	Constructs a new matrix from a 2D slice.
 */
func MatrixFromArray(x [][]float64) *Matrix {
	m := new(Matrix)
	m.Shape = []int{len(x), len(x[0])}
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
	m.Shape = []int{rows, columns}
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

/**
 *	Constructs a matrix of binomially distributed samples with the given dimensions.
 */
func MatrixBin(p float64, rows, columns int) *Matrix {
	rand.Seed(int64(time.Now().Unix()))

	// Useful functions
	max := uint64(15)
	getRand := func () uint64 { return uint64(rand.Int63()) % max }
	fact := func (n uint64) uint64 {   
		var val uint64 = 1
		var i uint64 = 1
		for ; i <= n; i++ {
			val *= i
		}
		return uint64(val)
	}
	ncr := func (n, r uint64) uint64 {
		return fact(n) / (fact(r) * fact(n - r))
	}

	m := MatrixFromDims(rows, columns)
	binFunc := func(_ float64) float64 {
		x, n := getRand(), max
		return float64(ncr(n, x)) * math.Pow(p, float64(x)) * math.Pow(1 - p, float64(n - x))
	}
	Vectorize(m, binFunc)

	return m
}

/**
 *	Constructs a matrix of uniformly distributed samples with the given dimensions.
 */
func MatrixUnif(p float64, rows, columns int) *Matrix {
	rand.Seed(int64(time.Now().Unix()))

	// Useful functions
	max := uint64(1000)
	cutoff := uint64(float64(max) * p)
	getRand := func () uint64 { return uint64(rand.Int63()) % max }
	m := MatrixFromDims(rows, columns)
	unifFunc := func(_ float64) float64 {
		if getRand() <= cutoff {
			return 1
		}
		return 0
	}
	Vectorize(m, unifFunc)

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
 *	Sets the matrix row at index i to the values provided.
 */
func SetRow(m *Matrix, i int, x []float64) {
	for j := 0; j < m.Shape[1]; j++ {
		Set(m, i, j, x[j])
	}
}

/**
 *	Sets the matrix column at index j to the values provided.
 */
func SetColumn(m *Matrix, j int, x []float64) {
	for i := 0; i < m.Shape[1]; i++ {
		Set(m, i, j, x[i])
	}
}

/**
 *	Gets the matrix value at the given coordinates i and j.
 */
func Get(m *Matrix, i, j int) float64 {
	return m.content[i][j]
}

/**
 *	Gets the row at index i.
 */
func GetRow(m *Matrix, i int) []float64 {
	return m.content[i]
}

/**
 *	Gets the column at index j.
 */
func GetColumn(m *Matrix, j int) []float64 {
	column := make([]float64, m.Shape[0])
	for i := 0; i < m.Shape[0]; i++ {
		column[i] = Get(m, i, j)
	}
	return column
}

/* MATH FUNCTIONS */

/**
 *	Adds two matrices and returns the resulting matrix.
 */
func Add(lhs *Matrix, rhs *Matrix) (Matrix, error) {
	// Ensure the matrices are summable
	if lhs.Shape[0] != rhs.Shape[0] || lhs.Shape[1] != rhs.Shape[1] {
		return *lhs, errors.New("Incorrect Shapes for adding: (" + string(lhs.Shape[0]) + ", " + string(lhs.Shape[1]) + ") and (" + string(rhs.Shape[0]) + ", " + string(rhs.Shape[1]) + ")")
	}

	// Create a new matrix with the same Shape and add lhs plus rhs
	res := MatrixFromDims(lhs.Shape[0], lhs.Shape[1])
	add := func(x float64, i, j int) float64 { return Get(lhs, i, j) + Get(rhs, i, j) }
	VectorizeWithDims(res, add)

	return *res, nil
}

/**
 *	Subtracts second matrix from the first and returns the resulting matrix.
 */
func Subtract(lhs *Matrix, rhs *Matrix) (Matrix, error) {
	// Ensure the matrices are compatible
	if lhs.Shape[0] != rhs.Shape[0] || lhs.Shape[1] != rhs.Shape[1] {
		return *lhs, errors.New("Incorrect Shapes for subtracting: (" + string(lhs.Shape[0]) + ", " + string(lhs.Shape[1]) + ") and (" + string(rhs.Shape[0]) + ", " + string(rhs.Shape[1]) + ")")
	}

	// Create a new matrix with the same Shape and add lhs plus rhs
	res := MatrixFromDims(lhs.Shape[0], lhs.Shape[1])
	add := func(x float64, i, j int) float64 { return Get(lhs, i, j) - Get(rhs, i, j) }
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
	if lhs.Shape[1] != rhs.Shape[0] {
		return *lhs, errors.New("Incorrect Shapes " + string(lhs.Shape[1]) + " and " + string(rhs.Shape[0]))
	}

	// Create a new matrix from the outter dims of the given matrices
	res := MatrixFromDims(lhs.Shape[0], rhs.Shape[1])

	// Perform matrix multiplication
	for i := 0; i < lhs.Shape[0]; i++ {
		for j := 0; j < rhs.Shape[1]; j++ {
			for k := 0; k < rhs.Shape[0]; k++ {
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
 *	Returns the sum of all components in the matrix.
 */
func Sum(m *Matrix) float64 {
	sum := float64(0)
	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			sum += Get(m, i, j)
		}
	}
	return sum
}

/**
 *	Transposes the matrix into a new matrix.
 */
func Transposed(m *Matrix) *Matrix {
	copy := MatrixCopy(m)
	new_m := MatrixFromDims(m.Shape[1], m.Shape[0])
	transposeFunc := func (x float64, i, j int) float64 { return Get(copy, j, i) }
	VectorizeWithDims(new_m, transposeFunc)
	return new_m
}

/**
 *	Performs the given function on all values in the matrix.
 */
func Vectorize(m *Matrix, f func(float64) float64) {
	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			Set(m, i, j, f(Get(m, i, j)))
		}
	}
}

/**
 *	Performs the given function on all values in the matrix, with provided dims.
 */
func VectorizeWithDims(m *Matrix, f func(float64, int, int) float64) {
	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			Set(m, i, j, f(Get(m, i, j), i, j))
		}
	}
}

/**
 *	Prints a matrix, row by row.
 */
func PrintMatrix(m *Matrix) {

	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			m_ij := Get(m, i, j)
			if m_ij >= 0 {
				fmt.Printf(" ")
			}
			fmt.Printf("%.5f", m_ij)
			if j != m.Shape[1] - 1 {
				fmt.Printf(" ")
			}
		}
		fmt.Printf("\n")
	}
}