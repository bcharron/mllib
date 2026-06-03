package mllib

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestIdentityApply(t *testing.T) {
	i := Identity

	expected := []float64{10.0, -1.0, 0.001}
	actual := []float64{
		i.Apply(10.0),
		i.Apply(-1.0),
		i.Apply(0.001),
	}

	assert.InEpsilonSlice(t, expected, actual, 0.00001, "Identity should return the same value")
}
