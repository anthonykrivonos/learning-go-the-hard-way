package layers

import (
	e "enums"
	m "matrix"
	math "math"
)

type ActivationLayer struct {
	activation e.Activation
	size int
}

func ActivationLayerInit(activation e.Activation, size int) *ActivationLayer {
	l := new(ActivationLayer)
	l.activation = activation
	l.size = size
	return l
}

func Activate(l *ActivationLayer, mat m.Matrix) m.Matrix {
	switch l.activation {
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
	switch l.activation {
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