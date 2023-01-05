package stats

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strconv"
	"time"
)

type BenchmarkLog struct {
	Timestamp   time.Time     `json:"timestamp"`
	Index       string        `json:"index"`
	Rows        int           `json:"rows"`
	MemoryUsage float32       `json:"memoryUsage"`
	Concurrent  int           `json:"concurrent"`
	TotalTime   time.Duration `json:"totalTime"`
	LoadTime    time.Duration `json:"loadTime"`
	Latency     time.Duration `json:"latency"`
	FloatBit    int           `json:"floatBit"`
	EpochNum    int           `json:"epochNum"`
}

func GetLatencyArrSorted(epoch int, dir string) (arr []int) {
	for i := 0; i < epoch; i++ {
		filename := dir + "test_" + strconv.Itoa(i) + ".json"
		jsonFile, err := os.Open(filename)
		if err != nil {
			fmt.Println(err)
		}
		defer jsonFile.Close()

		byteValue, _ := ioutil.ReadAll(jsonFile)

		var benchmarkLog BenchmarkLog

		json.Unmarshal(byteValue, &benchmarkLog)

		arr = append(arr, int(benchmarkLog.Latency))
	}
	sort.Ints(arr)
	return
}
func GetLoadArrSorted(epoch int, dir string) (arr []int) {
	for i := 0; i < epoch; i++ {
		filename := dir + "test_" + strconv.Itoa(i) + ".json"
		jsonFile, err := os.Open(filename)
		if err != nil {
			fmt.Println(err)
		}
		defer jsonFile.Close()

		byteValue, _ := ioutil.ReadAll(jsonFile)

		var benchmarkLog BenchmarkLog

		json.Unmarshal(byteValue, &benchmarkLog)

		arr = append(arr, int(benchmarkLog.LoadTime))
	}
	sort.Ints(arr)
	return
}

func GetPercentile(sortedArr []int, percentile float32, epoch int) (res int) {
	index := float32(epoch) * percentile
	indexRounded := int(math.Round(float64(index)))
	res = sortedArr[indexRounded-1]

	fmt.Printf("THE %dth PERCENTILE: %f ms\n", int(100-(percentile*100)), float64(res)/1000000)
	return
}

func GetAvg(sortedArr []int, parameter string) (avg float32) {
	var sum int
	for _, el := range sortedArr {
		sum += el
	}
	avg = float32(sum) / float32(len(sortedArr))

	fmt.Printf("The average %s: %f ms\n", parameter, avg/1000000)
	return
}
