package main

import (
	"fmt"
	e "enums"
	m "matrix"
	l "layers"
	math "math"
)

type NeuralNetwork struct {
	input_size int
	weights []m.Matrix
	biases []m.Matrix
	layers []l.ActivationLayer
	reg_layers map[int]l.RegularizationLayer
}

func NeuralNetworkInit(input_size int) *NeuralNetwork {
	nn := new(NeuralNetwork)
	nn.input_size = input_size
	nn.reg_layers = make(map[int]l.RegularizationLayer)
	nn.layers = append(nn.layers, *l.ActivationLayerInit(e.Linear, input_size))
	fmt.Printf("Initialized neural network with input size %d\n", input_size)
	return nn
}

func AddActivationLayer(nn *NeuralNetwork, layer *l.ActivationLayer) {
	input_size := nn.layers[len(nn.layers) - 1].Size
	// Randomize new weights and biases matrices
	weights_mat := m.MatrixStdNorm(input_size, layer.Size)
	biases_mat := m.MatrixStdNorm(layer.Size, 1)
	// Append weights, biases, and layer to the list
	nn.weights = append(nn.weights, *weights_mat)
	nn.biases = append(nn.biases, *biases_mat)
	nn.layers = append(nn.layers, *layer)

	// Verbosity
	fmt.Printf(" - Added activation layer of type '%s' and Size %d\n", layer.Activation, layer.Size)
}

func AddRegularizationLayer(nn *NeuralNetwork, layer *l.RegularizationLayer) {
	// Simply add the regularization layer to the regularization layer map
	reg_layer_idx := len(nn.layers) - 1
	nn.reg_layers[reg_layer_idx] = *layer

	// Verbosity
	fmt.Printf(" - Added regularization layer of type '%s'\n", layer.Regularization)
}

func Reset(nn *NeuralNetwork) {
	layer_count := len(nn.layers)

	// Create copies of weights and biases
	weights_copy := nn.weights
	biases_copy := nn.biases
	// Allocate space for weights and biases
	nn.weights = make([]m.Matrix, layer_count)
	nn.biases = make([]m.Matrix, layer_count)
	// Overwrite the weights and biases
	for i := 0; i < layer_count; i++ {
		nn.weights[i] = m.MatrixStdNorm(weights_copy[i].size[0], weights_copy[i].size[1])
		nn.biases[i] = m.MatrixStdNorm(biases_copy[i].size[0], 1)
	}
}

func feedforward(nn *NeuralNetwork, x m.Matrix) []m.Matrix, []m.Matrix {
	layer_count := len(nn.layers)

	// Feedforward vector
	a := MatrixCopy(x)

	// Outputs
	z_outputs := make([]m.Matrix, layer_count)
	// Activated outputs
	a_outputs := make([]m.Matrix, layer_count)
	a_outputs[0] = a

	// Perform feedforward
	for i := 0; i < layer_count; i++ {
		// Get the output at the given layer
		z = m.Add(m.Multiply(a, nn.weights[i]), nn.biases[i])
		z_outputs[i] = z
		// Activate the output
		a = l.Activate(z)
		a_outputs[i] = a
	}

	// Return the resulting outputs and activated outputs
	return z_outputs, a_outputs
}

func backpropagate(nn *NeuralNetwork, y m.Matrix, z_outputs, a_outputs []m.Matrix) {
	layer_count := len(nn.layers)
	num_outputs := y.shape[0]

	// Instantiate a list of errors for outputs at each layer
	dCdZ := make([]float64, layer_count)
	for i := 0; i < layer_count; i++ { dCdZ[i] = 0.0 }

	// Record the error in the last layer first
	dCdZ[layer_count - 1] = m.Multiply(m.Subtract(y, a_outputs[layer_count - 1]), l.Derivative(nn.layers[layer_count - 1], z_outputs[layer_count - 1]))

	// Get the derivatives of the cost function w.r.t. the weights and biases
	dCdw := make([]m.Matrix, layer_count)
	dCdb := make([]m.Matrix, layer_count)
	for i := 0; i < layer_count; i++ {
		// Derivative of the cost function w.r.t. weights
		dCdw[i] = Multiply(m.Transposed(a_outputs[i]), dCdZ[i]) / float64(num_outputs)
		// Derivative of the cost function w.r.t. biases
		dCdb[i] = Multiply(MatrixOfOnes(1, num_outputs), dCdZ[i]) / float64(num_outputs)
	}

	// Return the cost function derivatives
	return dCdw, dCdb
}

func Train(nn *NeuralNetwork, X m.Matrix, y m.Matrix, batch_size int, epochs int, lr float64) {
	layer_count := len(nn.layers)
	num_outputs := y.shape[0]

	// Loop over number of epochs
	for e := 0; e < epochs; e++ {
		fmt.Printf("Epoch %d:\n", e + 1)
		batch_count := 0
		i := 0
		for ; i < y.shape[0]; {
			batch_count += 1

			// Compute the size of the current batch
			current_batch_size := batch_size
			if y.shape[0] - i < batch_size {
				current_batch_size = y.shape[0] - i
			}

			// Store the training data within the batch
			batch_X := m.MatrixFromDims(current_batch_size, X.shape[1])
			batch_y := m.MatrixFromDims(current_batch_size, y.shape[1])
			for j := 0; j < current_batch_size; j++ {
				batch_X[j] = m.GetRow(X, i + j)
				batch_y[j] = m.GetRow(y, i + j)
			}

			// Feedforward
			z_outputs, a_outputs = feedforward(nn, batch_X)
			// Backpropagate
			dCdw, dCdb = backpropagate(nn, batch_y, z_outputs, a_outputs)

			// Function that regularizes the input x only if a regularization layer is present
			conditionalReg := func (layer_num int, x m.Matrix) m.Matrix {
				if layer, contains := nn.reg_layers[layer_num]; contains {
					return l.Regularize(l, x)
				}
				return x
			}

			// Update weights and biases
			for j := 0; j < layer_count; j++ {
				lr_by_dCdw = MatrixCopy(dCdw[j])
				ScalarMultiply(lr_by_dCdw, lr)
				lr_by_dCdb = MatrixCopy(dCdb[j])
				ScalarMultiply(lr_by_dCdb, lr)

				nn.weights[j] = conditionalReg(j, Add(nn.weights[j], lr_by_dCdw)[0])
				nn.biases[j] = conditionalReg(j, Add(nn.biases[j], lr_by_dCdb)[0])
			}

			// Calculate loss
			sub, err = Subtract(batch_y, a_outputs[layer_count - 1])[0]
			Vectorize(sub, func (x float64) float64 { return math.Pow(x, 2) })
			sum := m.Sum(sub)
			loss := sum / num_outputs
			
			// Verbosity
			fmt.Printf(" - Batch %d (loss: )\n", batch_count, loss)

			i += batch_size
		}
	}
}

func Classify(nn *NeuralNetwork, x m.Matrix) m.Matrix {
	// To classify, simply feed forward up until the very end and get the activated output
	layer_count := len(nn.layers)
	z_outputs, a_outputs = feedforward(nn, x)
	return a_outputs[layer_count - 1]
}


func main() {
	// // var a enums.LayerType = "Activation"
	// // fmt.Println(a)
	// // Test matrix multiplication

	// // Test 1
	// var m1 *m.Matrix = m.MatrixFromArray( [][]float64{{ 1, 2 }, { 3, 4 }} )
	// var m2 *m.Matrix = m.MatrixFromArray( [][]float64{{ 1, 0 }, { 0, 1 }} )
	// var m1_by_m2, e = m.Multiply(m1, m2)
	// if e != nil {
	// 	fmt.Println(e)
	// } else {
	// 	fmt.Printf("M1:\n")
	// 	m.PrintMatrix(m1)
	// 	fmt.Printf("\nM2:\n")
	// 	m.PrintMatrix(m2)
	// 	fmt.Printf("\nM1 * M2:\n")
	// 	m.PrintMatrix(&m1_by_m2)
	// 	fmt.Print("\n\n")
	// }


	// // Test 2
	// m1 = m.MatrixOfOnes(5, 5)
	// m2 = m.MatrixOfZeros(5, 5)
	// m1_by_m2, e = m.Multiply(m1, m2)
	// if e != nil {
	// 	fmt.Println(e)
	// } else {
	// 	fmt.Printf("M1:\n")
	// 	m.PrintMatrix(m1)
	// 	fmt.Printf("\nM2:\n")
	// 	m.PrintMatrix(m2)
	// 	fmt.Printf("\nM1 * M2:\n")
	// 	m.PrintMatrix(&m1_by_m2)
	// 	fmt.Print("\n\n")
	// }

	// // Test 3
	// m1 = m.MatrixStdNorm(5, 5)
	// m2 = m.MatrixIdentity(5, 5)
	// m1_by_m2, e = m.Multiply(m1, m2)
	// if e != nil {
	// 	fmt.Println(e)
	// } else {
	// 	fmt.Printf("M1:\n")
	// 	m.PrintMatrix(m1)
	// 	fmt.Printf("\nM2:\n")
	// 	m.PrintMatrix(m2)
	// 	fmt.Printf("\nM1 * M2:\n")
	// 	m.PrintMatrix(&m1_by_m2)
	// 	fmt.Print("\n\n")
	// }

	// // Test 4
	// m1 = m.MatrixStdNorm(5, 5)
	// m2 = m.MatrixCopy(m1)
	// m1_by_m2, e = m.Multiply(m1, m2)
	// if e != nil {
	// 	fmt.Println(e)
	// } else {
	// 	fmt.Printf("M1:\n")
	// 	m.PrintMatrix(m1)
	// 	fmt.Printf("\nM2:\n")
	// 	m.PrintMatrix(m2)
	// 	fmt.Printf("\nM1 * M2:\n")
	// 	m.PrintMatrix(&m1_by_m2)
	// 	fmt.Print("\n\n")
	// }

	// Test 5
	// m1 := m.MatrixUnif(.5, 5, 5)
	// m2 := m.MatrixCopy(m1)
	// m1_by_m2, e := m.Multiply(m1, m2)
	// if e != nil {
	// 	fmt.Println(e)
	// } else {
	// 	fmt.Printf("M1:\n")
	// 	m.PrintMatrix(m1)
	// 	fmt.Printf("\nM2:\n")
	// 	m.PrintMatrix(m2)
	// 	fmt.Printf("\nM1 * M2:\n")
	// 	m.PrintMatrix(&m1_by_m2)
	// 	fmt.Print("\n\n")
	// }

	// fmt.Printf("Original:\n")
	// var mat *m.Matrix = m.MatrixFromArray( [][]float64{{ 1, 2 }, { 3, 4 }} )
	// m.PrintMatrix(mat)

	// fmt.Printf("Sigmoid:\n")
	// var layer = l.ActivationLayerInit("Sigmoid", 32)
	// var res m.Matrix = l.Activate(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("ReLu:\n")
	// layer = l.ActivationLayerInit("ReLu", 32)
	// res = l.Activate(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("Linear:\n")
	// layer = l.ActivationLayerInit("Linear", 32)
	// res = l.Activate(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("Softmax:\n")
	// layer = l.ActivationLayerInit("Softmax", 32)
	// res = l.Activate(layer, *mat)
	// m.PrintMatrix(&res)


	// fmt.Printf("\nDerivatives:\n\n")

	// fmt.Printf("Sigmoid:\n")
	// layer = l.ActivationLayerInit("Sigmoid", 32)
	// res = l.Derivative(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("ReLu:\n")
	// layer = l.ActivationLayerInit("ReLu", 32)
	// res = l.Derivative(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("Linear:\n")
	// layer = l.ActivationLayerInit("Linear", 32)
	// res = l.Derivative(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("Softmax:\n")
	// layer = l.ActivationLayerInit("Softmax", 32)
	// res = l.Derivative(layer, *mat)
	// m.PrintMatrix(&res)

	// Regularization tests
	// fmt.Printf("Original:\n")
	// var mat *m.Matrix = m.MatrixStdNorm(5, 5)
	// m.PrintMatrix(mat)

	// fmt.Printf("Dropout:\n")
	// var layer = l.RegularizationLayerInit("Dropout", 0.5)
	// var res m.Matrix = l.Regularize(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("L1:\n")
	// layer = l.RegularizationLayerInit("L1", 0.2)
	// res = l.Regularize(layer, *mat)
	// m.PrintMatrix(&res)

	// fmt.Printf("L2:\n")
	// layer = l.RegularizationLayerInit("L2", 0.2)
	// res = l.Regularize(layer, *mat)
	// m.PrintMatrix(&res)

	nn := NeuralNetworkInit(32)
	AddActivationLayer(nn, l.ActivationLayerInit(e.ReLu, 16))
	AddActivationLayer(nn, l.ActivationLayerInit(e.ReLu, 8))
	AddRegularizationLayer(nn, l.RegularizationLayerInit(e.Dropout, 0.5))
	AddActivationLayer(nn, l.ActivationLayerInit(e.Softmax, 4))

}