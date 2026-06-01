package mllib

import (
	"fmt"
)

type ActivationFunction func(float32) float32

type Layer struct {
	Inputs     int
	Outputs    int
	W          []float32
	B          []float32
	Activation ActivationFunction
}

func NewLayer(inputs, outputs int, activation ActivationFunction) *Layer {
	layer := &Layer{
		Inputs:     inputs,
		Outputs:    outputs,
		W:          make([]float32, inputs*outputs),
		B:          make([]float32, outputs),
		Activation: activation,
	}

	return layer
}

func (layer *Layer) Compute(dst, src []float32) {
	if len(src) != layer.Inputs || len(dst) != layer.Outputs {
		s := fmt.Sprintf("src or dst do not match layer shape: %v != %v or %v != %v", len(src), layer.Inputs, len(dst), layer.Outputs)
		panic(s)
	}

	for i := 0; i < layer.Outputs; i++ {
		weights := layer.W[layer.Inputs*i : layer.Inputs*i+layer.Inputs]
		output := dotProduct(src, weights)
		dst[i] = layer.Activation(output + layer.B[i])
	}
}

func (layer *Layer) weightAtIdx(inputIdx, outputIdx int) float32 {
	return layer.W[layer.Inputs*outputIdx+inputIdx]
}

func dotProduct(m1, m2 []float32) float32 {
	if len(m1) != len(m2) {
		panic("matrices shapes do not match")
	}

	total := float32(0)

	for i := range len(m1) {
		total += m1[i] * m2[i]
	}

	// fmt.Printf("dot(%+v, %+v) = %v\n", m1, m2, total)

	return total
}
