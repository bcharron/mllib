package mllib

import "math/rand"

func randRange(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
