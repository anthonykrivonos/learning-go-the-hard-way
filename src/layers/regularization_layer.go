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