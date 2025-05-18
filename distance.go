package KMeans

import (
	"math"
)

func L1[T Number](a, b Vector[T]) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(float64(a[i] - b[i]))
	}
	return sum
}

func L2[T Number](a, b Vector[T]) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Pow(float64(a[i]-b[i]), 2)
	}
	return math.Sqrt(sum)
}

func L1_fp32(a, b Vector[float32]) float32 {
	sum := float32(0)
	for i := range a {
		sum += float32(math.Abs(float64(a[i] - b[i])))
	}
	return sum
}

func L2_fp32(a, b Vector[float32]) float32 {
	sum := float32(0)
	for i := range a {
		sum += float32(math.Pow(float64(a[i]-b[i]), 2))
	}
	return float32(math.Sqrt(float64(sum)))
}

var _ Distance[float32, float32] = L1_fp32
var _ Distance[float32, float32] = L2_fp32

func L1_f64(a Vector[float64], b Vector[float64]) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

func L2_f64(a Vector[float64], b Vector[float64]) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Pow(a[i]-b[i], 2)
	}
	return math.Sqrt(sum)
}

var _ Distance[float64, float64] = L1_f64
var _ Distance[float64, float64] = L2_f64
