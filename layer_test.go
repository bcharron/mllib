package mllib

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewLayer(t *testing.T) {
	layer := NewLayer(2, 3, Identity)

	assert.Equal(t, 2, layer.Inputs)
	assert.Equal(t, 3, layer.Outputs)
	assert.Equal(t, 2*3, len(layer.W))
	assert.Equal(t, 3, len(layer.B))
}

func TestCompute(t *testing.T) {
	layer := NewLayer(2, 3, Identity)

	inputs := []float32{1.0, 2.0}

	layer.W = []float32{1.0, 2.0, 1.0, 2.0, 1.0, 2.0}
	layer.B = []float32{0.0, 0.0, 0.0}

	outputs := make([]float32, 3)

	layer.Compute(outputs, inputs)

	assert.InEpsilonSlice(t, []float32{5.0, 5.0, 5.0}, outputs, 0.00001)
}
