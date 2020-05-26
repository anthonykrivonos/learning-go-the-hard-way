<script src="//yihui.org/js/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async
  src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Learning Go the Hard Way

### A.K.A. coding a neural network in Go.

> Nothing in life should come easy. I taught myself Go(lang) by rebuilding the Multi-layer Perceptron (MLP) neural net I coded a few months ago [in Python](https://github.com/anthonykrivonos/mlp).

My notes and findings from my journey, in tutorial-style, are written in this README. Everything else can be found in the code.

## Installing and Running

```
git clone {this repo} && ./setpath.sh
cd learning-go-the-hard-way
./setpath.sh
```

## Tutorial

### 1. Creating a New Project

Navigate to an empty folder and run the following to create a source directory and set your `GOPATH` to it (if you're on Max/UNIX).

```
mkdir src && export GOPATH=$PWD && cd src
```

> I'll admit `GOPATH` is one of the stupidest and most annoying things I've encountered thus far throughout this tutorial and have been on the verge of even coding a new project in C due to it.

We'll continue by creating a new Go module in the `/src/mlp` directory.

```
mkdir mlp && cd mlp && go mod init mlp`
```

Next, we'll create a new file for the neural network and add some boilerplate to it.

By convention, files are underscore-cased.
```
touch neural_net.go
```

Let's add some boilerplate to this file:
```
// Define the MLP package in the root
package mlp

import "fmt"

func main() {
	// our code goes here
	fmt.Println("this is where our code will go")
}
```

Sanity check––compile and run the code via the following line.
```
go run neural_net.go
```
> `this is where our code will go`

After noticing the pun in the output, I decided to create the `Loss`, `LayerType`, `Regularization`, and `Activation` enum types first. These enums will help dictate how each section of our neural network will operate, and will teach us both about enums and importing modules into a main module.

### 2. Creating Enum Submodules

We'll start off by creating 3 new modules the same way we did before.

```
cd ../
mkdir enums && cd enums
go mod init enums
touch loss.go
touch layer_type.go
touch regularization.go
touch activation.go
```

Writing an enumeration is pretty easy but convoluted in Go. We'll declare the `Loss` enum as follows (below are the contents of `src/enums/loss.go`).

```
package enums

type Loss string

const (
    CrossEntropy = "CrossEntropy"
    MSE = "MSE"
)
```

Note that we aren't importing `"fmt"`. This is because we don't need any input scanning or printing. This is what [Go's documentation](https://golang.org/pkg/fmt/) says about the `fmt` package:

> Package fmt implements formatted I/O with functions analogous to C's printf and scanf. The format 'verbs' are derived from C's but are simpler.

Now, try writing the rest of the enums in the same fashion (enum variable name matches its value). Here are the values each respective enum contains:

#### Enums to Declare:
1. LayerType
    - Activation
    - Regularization
2. Regularization
    - Dropout
    - L1
    - L2
3. Activation
    - Sigmoid
    - ReLu
    - Linear
    - Softmax

Once you do this, you'll notice we've come across our first errors! They look similar to this one:

> Activation redeclared in this block previous declaration at ./activation.go:5:6

It seems that enums are treated as though they are from the same scope, even though they pertain to different enum definitions. We'll alleviate this issue by changing the following enum value names:

```
mlp/enums/layer_type.go:
    Activation -> ActivationLayer
    Regularization -> RegularizationLayer
```

Finally! Let's try importing the `Activation` enum into `neural_net.go`. The following should give the corresponding output.

```
package main

import (
	"fmt"
	"enums"
)

func main() {
	var a enums.LayerType = "Activation"
	fmt.Println(a)
}
```

> `> Activation`

We're now ready for more exciting stuff.

### 3. Coding Matrix Class

Go has a pretty robust numpy-esque library called [Gonum](https://www.gonum.org/), but its confusing documentation and intimidating scale led me to create a `Matrix` class from the ground up that allows for matrices to be instantiated, added, multiplied, etc. This'll be the closest thing we'll get to numpy here.

Create a new package under `/src/matrix` called `matrix` using the instructions from step (2) and then create a new file called `matrix.go` in that directory.

We'll start by defining the package and importing a few things that'll be useful later.

```
package matrix

import (
	"fmt"
	"errors"
	rand "math/rand"
	"time"
)
```

Now, let's define the `Matrix` class.

```
type Matrix struct {
	Shape []int
	content [][]float64
}
```

If you look not-so-closely, you'll notice that this isn't a class but, in fact, a C-like struct. From my countless hours of Googling around, I can safely say that Google has made Go a very opinionated language. We'll thus have to cater to its quirks and create factory methods that imitate constructors, as follows.

```
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
```

Whereas the code is pretty self-explanatory, the first function emulates a copy constructor by taking in a pointer to a matrix, creating a new `Matrix` struct, copying over the properties of the matrix by value, and finally returning a pointer to the new copy. You might think we're simply pointing the properties in the new matrix to the properties in the old matrix, rather than copying them and that's what I thought too, at first. However, this is simply Go's rather confusing but simple copy syntax. You never know whether you're abiding to old school programming paradigms or modern ones at this rate.

As an aside, the `make` function in `MatrixFromDims` works like `malloc` in the sense that it allocates space for the first and second dimensions of our matrix. The array, or "slice", syntax takes a bit of getting used to but is fairly intuitive.

Now, let's write couple methods that you're probably already familiar with... getters and setters! Once again, these will have to be global methods that take in a matrix by value as the first argument.

```
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
```

Before we head straight on over to the math functions, let's create two methods that'll simplify matrix operations for us in the future. `Vectorize` is simply a closure, or a callback function (you can think of these as function pointers if you're not familiar with Node.js/Swift) that sends a value from the matrix as its only argument and expects you to return a modified version of that value. Don't worry if that's confusing, the upcoming examples will surely clear it up for you. Additionally, you can look at [`numpy.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html), since this is the effect we're going for.

```
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
```

`VectorizeWithDims` is simply an extension of `Vectorize` that also passes in the indices of the row and column in which the current value was found in the matrix. To summarize, these functions allow us to call a manipulation function across every component in the matrix.

Now, let's put these functions and our `Get` function to good use with these straightforward scalar addition and multiplication methods.

```
/**
 *	Adds a scalar to all values in the matrix.
 */
func ScalarAdd(m *Matrix, s float64) {
	addS := func(x float64) float64 { return x + s }
	Vectorize(m, addS)
}

/**
 *	Multiplies all values in the matrix by a scalar.
 */
func ScalarMultiply(m *Matrix, s float64) {
	multS := func(x float64) float64 { return x * s }
	Vectorize(m, multS)
}
```

In the above functions, we simply define a subfunction that performs either the additive or multiplicative manipulation using $s$ on $x$, and then perform the vectorize function. Obviously, we aren't returning anything since we're modifying the matrix `m`. Using `VectorizeWithDims`, we create the matrix addition and subtraction functions as follows.

```
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
```

Note that we return an error if the matrices being added are not the same Shape. Additionally, instead of modifying a matrix that's effectively passed by reference, we are creating a new `res` matrix that holds the sum of `lhs` and `rhs`. Matrix multiplication is a little more tricky. Since the goal of this tutorial is to learn Go and build a neural network in the process, I settled for a simple $O(n^3)$ matrix multiplication implementation.

```
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
```

Visit [here](https://www.tutorialspoint.com/matrix-multiplication-algorithm) for more information on this algorithm.

Finally, let's add a `Tranpose` function using `VectorizeWithDims` that simply transposes the matrix into a new one (`(i, j) -> (j, i)`), as well as a function called `Sum` that simply returns the sum of all components in the matrix.
```
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
```

We're almost done with our `Matrix` class! I simply added a few extra constructor-like functions to make declaring special matrices easier in the future. I wish I had more time to explain them in detail, but I hope they are all pretty self-explanatory.

```
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
```

Phew! That was a lot but now we've got a custom Matrix class that we can use to store weights in biases between each layer in our future neural network!


### 4. Let the Neural Network Begin: Coding the ActivationLayer Class

Now, we'll first create a new package under `/src/layers` called `layers` using the instructions from step (2) and then create a new file called `activation_layer.go` in that directory.

Now, we'll use `Vectorize` to implement 4 activation functions and their derivatives. You
can learn more about these functions in great detail through ML research papers and I'll
simply cover the gist of each one.

#### Sigmoid Activation
Used to penalize large-magnitude outputs.

`$ Sigmoid(x) = \frac{1}{1 + e^{-x}} $`

Derivative used for backpropagation:

`$ SigmoidDeriv(x) = Sigmoid(x) \times (1 - Sigmoid(x)) $`

#### ReLU Activation
Zeros out negative outputs.

`$ ReLu(x) = { x if x > 0 else 0 } $`

Derivative used for backpropagation:

`$ ReLuDeriv(x) = { 1 if x > 0 else 0 } $`

#### Softmax Activation
Used mainly for one-hot-encoded outputs, putting most weight on
the probable output.

`$ Softmax(x) = e^{x}_j / \sum_{k}e^{x}_k, \text{where j is the index of the target one-hot-encoded features and the denominator loops over all one-hot-encoded features k} $`

Derivative used for backpropagation:

`$ SoftmaxDeriv(x) = Softmax(x) \times (1 - Softmax(x)) $`

Now, let's define the `ActivationLayer` struct as follows.

```
package layers

import (
	e "enums"
	m "matrix"
	math "math"
)

type ActivationLayer struct {
	Activation e.Activation
	Size int
}
```

Now that we've written the mathematical equations for the activation functions, we'll implement the `Activate` and `Derivative` functions below, along with a basic `ActivationLayerInit` constructor.

```
func ActivationLayerInit(Activation e.Activation, Size int) *ActivationLayer {
	l := new(ActivationLayer)
	l.Activation = Activation
	l.Size = Size
	return l
}

func Activate(l *ActivationLayer, mat m.Matrix) m.Matrix {
	switch l.Activation {
		case e.Sigmoid:
			sigmoidFunc := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
			m.Vectorize(&mat, sigmoidFunc)
		case e.ReLu:
			reluFunc := func(x float64) float64 {
				if x > 0 {
					return x
				}
				return 0
			}
			m.Vectorize(&mat, reluFunc)
		case e.Softmax:
			softmaxFunc := func(x float64) float64 {
				num := math.Exp(x)
				denom := 0.0
				for i := 0; i < mat.Shape[0]; i++ {
					for j := 0; j < mat.Shape[1]; j++ {
						denom += math.Exp(m.Get(&mat, i, j))
					}
				}
				return num / denom
			}
			m.Vectorize(&mat, softmaxFunc)
		default:
			break
	}
	return mat
}
	
func Derivative(l *ActivationLayer, mat m.Matrix) m.Matrix {
	switch l.Activation {
		case e.Sigmoid:
			sig := Activate(l, mat)
			subtracted, err := m.Subtract(m.MatrixOfOnes(mat.Shape[0], mat.Shape[1]), &sig)
			if err == nil {
				mat, err = m.Multiply(&sig, &subtracted)
			} else {
				return mat
			}
		case e.ReLu:
			reluFunc := func(x float64) float64 {
				if x > 0 {
					return 1
				}
				return 0
			}
			m.Vectorize(&mat, reluFunc)
		case e.Softmax:
			smax := Activate(l, mat)
			subtracted, err := m.Subtract(m.MatrixOfOnes(mat.Shape[0], mat.Shape[1]), &smax)
			if err == nil {
				mat, err = m.Multiply(&smax, &subtracted)
			} else {
				return mat
			}
		default:
			mat = *m.MatrixOfOnes(mat.Shape[0], mat.Shape[1])
	}
	return mat
}
```

### 5. Let the Neural Network Begin: Coding the RegularizationLayer Class

We'll reuse a lot of the techniques from step (4) in order to create a new type of layer, the regularization layer, which performs some kind of manipulation on the weights between activation layers.

We define the `RegularizationLayer` struct in a similar fashion as `ActivationLayer`.

```
package layers

import (
	e "enums"
	m "matrix"
	math "math"
	rand "math/rand"
	"time"
)

type RegularizationLayer struct {
	Regularization e.Regularization
	parameter float64
}

func RegularizationLayerInit(Regularization e.Regularization, parameter float64) *RegularizationLayer {
	l := new(RegularizationLayer)
	l.Regularization = Regularization
	l.parameter = parameter
	return l
}
```

`regularization` stores the type of regularization this layer will perform and the `parameter` stores either the dropout rate for Dropout Regularization or `$\lambda$` for L1 and L2 regularization.

Next, we'll implement these regularizations in the `Regularize` function. Once again, I'm a little lazy and tired to bring in the mathematical background here, but the internet is your oyster in researching more about regularization in machine learning.

```
func Regularize(l *RegularizationLayer, weights m.Matrix) m.Matrix {
	switch l.Regularization {
		case e.Dropout:
			dropout_rate := l.parameter

			// Seed the random generator
			rand.Seed(int64(time.Now().Unix()))
			max := uint64(1000)
			cutoff := uint64(float64(max) * dropout_rate)
			getRand := func () uint64 { return uint64(rand.Int63()) % max }

			// Randomly drop weight components
			dropoutFunc := func (x float64) float64 {
				if getRand() <= cutoff {
					return x
				}
				return 0
			}
			m.Vectorize(&weights, dropoutFunc)
		case e.L1:
			l1_lambda := math.Abs(l.parameter)
			l1Func := func (x float64) float64 {
				if x >= 0 {
					return 1 / l1_lambda
				}
				return -1 / l1_lambda
			}
			l1_weights := m.MatrixCopy(&weights)
			m.Vectorize(l1_weights, l1Func)
			weights, _ = m.Subtract(&weights, l1_weights)
		case e.L2:
			l2_lambda := math.Abs(l.parameter)
			l2_weights := m.MatrixCopy(&weights)
			m.ScalarMultiply(l2_weights, 1 / l2_lambda)
			weights, _ = m.Subtract(&weights, l2_weights)
		default:
			break
	}
	return weights
}
```

Now, we have all the basic building blocks in place and are ready to code the actual `NeuralNetwork` class.

## References

## Author

Anthony Krivonos ([GitHub](https://github.com/anthonykrivonos) | [Portfolio](https://anthonykrivonos.com))

