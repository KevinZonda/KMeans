package KMeans

import (
	"math/rand/v2"
	"reflect"

	"golang.org/x/exp/constraints"
)

type Task[T Number, R constraints.Float] struct {
	K        int
	Data     []Vector[T]
	Delta    float64
	Shuffle  bool
	Distance Distance[T, R]

	centroids []Vector[T]

	clusters        []Cluster[T]
	UseAccumForMean bool
}

func (t *Task[T, R]) Run() []Vector[T] {
	if t.K <= 0 {
		return nil
	}

	if t.K > len(t.Data) {
		return t.Data
	}

	if t.Shuffle {
		rand.Shuffle(len(t.Data), func(i, j int) {
			t.Data[i], t.Data[j] = t.Data[j], t.Data[i]
		})
	}

	t.centroids = t.kmeanspp_init()
	t.clusters = make([]Cluster[T], t.K)
	lastCentroids := t.centroids
	for {
		t.clearClusters(t.centroids)
		t.Assign()
		t.centroids = t.UpdateCentroids(t.clusters)
		if reflect.DeepEqual(lastCentroids, t.centroids) {
			break
		}
		lastCentroids = t.centroids
	}
	return t.centroids
}

func (t *Task[T, R]) kmeanspp_init() []Vector[T] {
	centroids := make([]Vector[T], 1)
	centroids[0] = t.Data[rand.IntN(len(t.Data))]

	for k := 1; k < t.K; k++ {
		distances := make([]float64, len(t.Data))
		var sumDistances float64

		for i, point := range t.Data {
			minDist := float64(t.Distance(point, centroids[0]))
			for j := 1; j < len(centroids); j++ {
				dist := float64(t.Distance(point, centroids[j]))
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist * minDist // Square the distance
			sumDistances += distances[i]
		}

		// Select next centroid with probability proportional to squared distance
		r := rand.Float64() * sumDistances
		var cumSum float64
		var nextCentroid Vector[T]

		for i, point := range t.Data {
			cumSum += distances[i]
			if cumSum >= r {
				nextCentroid = point
				break
			}
		}

		centroids = append(centroids, nextCentroid)
	}

	return centroids
}

func (t *Task[T, R]) clearClusters(centroids []Vector[T]) {
	for i := range t.clusters {
		t.clusters[i].Centroid = centroids[i]
		t.clusters[i].Points = nil
	}
}

func (t *Task[T, R]) Assign() {
	distances := initMatrix[R](t.K, len(t.Data))

	for i, point := range t.Data {
		for j, centroid := range t.centroids {
			distances[j][i] = t.Distance(point, centroid)
		}
	}
	// transpose the matrix
	distances = transpose(distances)

	for i, point := range t.Data {
		minDistIndex := argmin(distances[i])
		t.clusters[minDistIndex].Points = append(
			t.clusters[minDistIndex].Points,
			point,
		)
	}
}

func (t *Task[T, R]) UpdateCentroids(clusters []Cluster[T]) []Vector[T] {
	centroids := make([]Vector[T], len(clusters))
	for i, cluster := range clusters {
		if len(cluster.Points) == 0 {
			centroids[i] = cluster.Centroid
			continue
		}

		centroids[i] = t.mean(cluster.Points)
	}
	return centroids
}

func (t *Task[T, R]) mean(points []Vector[T]) Vector[T] {
	mean := make(Vector[T], len(points[0]))
	length := T(len(points))
	if t.UseAccumForMean {
		for _, point := range points {
			for j := range point {
				mean[j] += (point[j] / length)
			}
		}
	} else {
		for _, point := range points {
			for j := range point {
				mean[j] += point[j]
			}
		}
		for j := range mean {
			mean[j] /= length
		}
	}
	return mean
}

func argmin[T Number](slice []T) int {
	min := slice[0]
	minIndex := 0
	for i, v := range slice {
		if v < min {
			min = v
			minIndex = i
		}
	}
	return minIndex
}

func transpose[T Number](matrix [][]T) [][]T {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return nil
	}

	rows, cols := len(matrix), len(matrix[0])
	transposed := make([][]T, cols)
	data := make([]T, rows*cols)
	for i := range transposed {
		transposed[i] = data[i*rows : (i+1)*rows]
	}

	for i := 0; i < rows*cols; i++ {
		row, col := i/cols, i%cols
		data[col*rows+row] = matrix[row][col]
	}

	return transposed
}

func initMatrix[T Number](m, n int) [][]T {
	matrix := make([][]T, m)
	for i := 0; i < m; i++ {
		matrix[i] = make([]T, n)
	}
	return matrix
}
