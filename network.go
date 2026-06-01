package mllib

import (
	"fmt"
)

type Network struct {
	Layers []*Layer
}

func NewNetwork(layerSize ...int) *Network {
	net := &Network{
		Layers: make([]*Layer, len(layerSize)-1),
	}

	for idx, size := range layerSize[:len(layerSize)-1] {
		next := layerSize[idx+1]
		fmt.Printf("Layer[%d] = %v -> %v\n", idx, size, next)
		net.Layers[idx] = NewLayer(size, next, Identity)
	}

	fmt.Printf("Layers: %+v\n", net.Layers)

	return net
}

func (n *Network) LoadWeights(filename string) {

}

func (n *Network) SaveWeights(filename string) {
}

func (n *Network) lastLayer() *Layer {
	lastLayerNb := len(n.Layers) - 1
	return n.Layers[lastLayerNb]
}

func (n *Network) Forward(dst, src []float32) {
	if len(dst) != n.lastLayer().Outputs {
		s := fmt.Sprintf("Shape of output does not match: Expected %v but got %v\n", n.lastLayer().Outputs, len(dst))
		panic(s)
	}

	if len(src) != n.Layers[0].Inputs {
		panic("Shape of input does not match")
	}

	input := src
	var outputs []float32

	for x := range len(n.Layers) - 1 {
		layer := n.Layers[x]
		outputs = make([]float32, layer.Outputs)
		fmt.Printf("Layer[%v].Forward(%+v, %+v)\n", x, input, outputs)
		layer.Forward(outputs, input)

		input = outputs
	}

	n.lastLayer().Forward(dst, outputs)
}
