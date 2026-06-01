package mllib

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewNetwork(t *testing.T) {
	nn := NewNetwork(2, 3, 2)

	assert.Equal(t, 2, len(nn.Layers))
	assert.Equal(t, 2, nn.Layers[0].Inputs)
	assert.Equal(t, 3, nn.Layers[0].Outputs)
	assert.Equal(t, 3, nn.Layers[1].Inputs)
	assert.Equal(t, 2, nn.Layers[1].Outputs)

	for _, layer := range nn.Layers {
		for x := range len(layer.W) {
			layer.W[x] = float32(x)
		}

		for x := range len(layer.B) {
			layer.B[x] = float32(x)
		}
	}

	// Layer 1:
	// [1.0 2.0]T . [0 1 2 3 4 5] = [2 9 16]
	// 1: 1*0+2*1 + 0 = 2
	// 2: 1*2+2*3 + 1 = 9
	// 3: 1*4+2*5 + 2 = 16

	// Layer 2:
	// [2 9 16]T . [0 1 2 3 4 5] = [41 123]
	// 1: 2*0 + 9*1 + 16*2 + 0 = 41
	// 2: 2*3 + 9*4 + 16*5 + 1 = 123

	inputs := []float32{1.0, 2.0}
	outputs := make([]float32, 2)
	nn.Compute(outputs, inputs)

	fmt.Printf("Outputs: %v\n", outputs)
	assert.InEpsilonSlice(t, []float32{41.0, 123.0}, outputs, 0.00001)
}

