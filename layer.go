package mllib

import (
	"fmt"
)

type ActivationFunction func(float64) float64

type Layer struct {
	Inputs     int
	Outputs    int
	W          []float64
	B          []float64
	Activation Activation
}

func NewLayer(inputs, outputs int, activation Activation) *Layer {
	layer := &Layer{
		Inputs:     inputs,
		Outputs:    outputs,
		W:          make([]float64, inputs*outputs),
		B:          make([]float64, outputs),
		Activation: activation,
	}

	return layer
}

func (layer *Layer) Forward(dst, src []float64) {
	if len(src) != layer.Inputs || len(dst) != layer.Outputs {
		s := fmt.Sprintf("src or dst do not match layer shape: %v != %v or %v != %v", len(src), layer.Inputs, len(dst), layer.Outputs)
		panic(s)
	}

	for i := 0; i < layer.Outputs; i++ {
		base := layer.Inputs * i
		weights := layer.W[base : base+layer.Inputs]
		output := dotProduct(src, weights)
		dst[i] = layer.Activation.Apply(output + layer.B[i])
	}
}

func (layer *Layer) weightAtIdx(inputIdx, outputIdx int) float64 {
	return layer.W[layer.Inputs*outputIdx+inputIdx]
}

func dotProduct(m1, m2 []float64) float64 {
	if len(m1) != len(m2) {
		panic("matrices shapes do not match")
	}

	total := float64(0)

	for i := range len(m1) {
		total += m1[i] * m2[i]
	}

	// fmt.Printf("dot(%+v, %+v) = %v\n", m1, m2, total)

	return total
}
