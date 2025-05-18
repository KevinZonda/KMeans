package KMeans

import (
	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Integer | constraints.Float
}

type Vector[T Number] []T

type Distance[T Number, R constraints.Float] func(a, b Vector[T]) R

type Cluster[T Number] struct {
	Centroid Vector[T]
	Points   []Vector[T]
}

type DistancedCluster[T Number, R constraints.Float] struct {
	Centroid Vector[T]
	Points   []VectorWithDistance[T, R]
}

type VectorWithDistance[T Number, R constraints.Float] struct {
	Vector   Vector[T]
	Distance R
}
