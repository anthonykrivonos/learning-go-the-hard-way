package layers

import (
	"enums"
	m "matrix"
)

type ActivationLayer struct {
	size int16
	activation string
	activation_parameter float32
	layer_type string
}
const (
    Sigmoid = "Sigmoid"
    ReLu = "ReLu"
    LeakyReLu = "LeakyReLu"
    NoisyReLu = "NoisyReLu"
    ELU = "ELU"
    Linear = "Linear"
    Softmax = "Softmax"
)
	
func (l *ActivationLayer) activate(x []float64) float64 {
	switch l.activation {
		case Activation.Sigmoid:
			return 
		default:
			return x
	}
}