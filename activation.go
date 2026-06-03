package mllib

import (
	"math"
)

type Activation interface {
	Apply(x float64) float64      // x is the output of Wx+b
	Derivative(y float64) float64 // y is the output of a previous Apply()
}

var (
	Identity = identity{}
	Tanh     = tanh{}
)

type identity struct{}

type tanh struct{}

func (identity) Apply(x float64) float64 {
	return x
}

func (identity) Derivative(y float64) float64 {
	return 1
}

func (tanh) Apply(x float64) float64 {
	return math.Tanh(x)
}

func (tanh) Derivative(y float64) float64 {
	// return 1.0 - math.Pow(math.Tanh(x), 2)
	return 1.0 - (y * y)
}
