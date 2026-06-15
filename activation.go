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
	Relu     = relu{}
)

type identity struct{}

type tanh struct{}

type relu struct{}

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
	return 1.0 - (y * y)
}

func (relu) Apply(x float64) float64 {
	return math.Max(0, x)
}

func (relu) Derivative(x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return 1
	}
}
