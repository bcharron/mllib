package mllib

import (
	"fmt"
)

type Network struct {
	Layers []*Layer
}

func NewNetwork(activation Activation, layerSize ...int) *Network {
	net := &Network{
		Layers: make([]*Layer, len(layerSize)-1),
	}

	lastElementIdx := len(net.Layers) - 1

	for idx, size := range layerSize[:len(layerSize)-1] {
		next := layerSize[idx+1]
		fmt.Printf("Layer[%d] = %v -> %v\n", idx, size, next)

		if idx == lastElementIdx {
			fmt.Printf("Layer %d activation is Identity\n", idx)
			activation = Identity
		}

		net.Layers[idx] = NewLayer(size, next, activation)
	}

	fmt.Printf("Layers: %+v\n", net.Layers)

	return net
}

func (n *Network) RandomizeWeights() {
	for _, layer := range n.Layers {
		layer.RandomizeWeights()
	}
}

func (n *Network) LoadWeights(filename string) {

}

func (n *Network) SaveWeights(filename string) {
}

func (n *Network) lastLayer() *Layer {
	lastLayerNb := len(n.Layers) - 1
	return n.Layers[lastLayerNb]
}

func (n *Network) Forward(dst, src []float64) {
	if len(dst) != n.lastLayer().Outputs {
		s := fmt.Sprintf("Shape of output does not match: Expected %v but got %v\n", n.lastLayer().Outputs, len(dst))
		panic(s)
	}

	if len(src) != n.Layers[0].Inputs {
		panic("Shape of input does not match")
	}

	input := src
	var outputs []float64

	for x := range len(n.Layers) - 1 {
		layer := n.Layers[x]
		outputs = make([]float64, layer.Outputs)
		fmt.Printf("Layer[%v].Forward(%+v, %+v)\n", x, input, outputs)
		layer.Forward(outputs, input)

		input = outputs
	}

	n.lastLayer().Forward(dst, outputs)
}

func (n *Network) Backward(dst, expected []float64) {
	// WIP
	for idx := len(n.Layers) - 1; idx > 0; idx-- {
		layer := n.Layers[idx]

		loss := make([]float64, layer.Outputs)

		for x := 0; x < layer.Outputs; x++ {
			err := layer.cache[x] - expected[x]
			mse := err * err
			loss[x] = mse
		}
	}
}

func (n *Network) Mutate(probability, rate float64) {
	for idx, layer := range n.Layers {
		layer.Mutate(probability, rate)
	}
}
