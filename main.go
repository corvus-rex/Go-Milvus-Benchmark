package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"gomilvus/async"
	"gomilvus/stats"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/milvus"
)

var collectionName = "benchmarking1"

// total 300000 rows (segment * loop * cluster)
// Insertion process is divided into cluster > loop > segment to avoid gRPC message size limit
// Insertion process is divided this way to avoid gRPC maximum message limit
var initialRow = 10 // amount of rows in pre-generated data from CSV
var segment = 10    // number of rows per insertion
var loop = 3000     // number of segments per Populate operation
var cluster = 10    // number of Populate operation per main()

var searchEpoch = 500 // number of search iteration

var dir = "./Benchmark/GPU_IVF_Flat/300K/"       // location where benchmarking results are saved
var benchmarkDataset = "./benchmark_dataset.csv" // name of dataset where pre-generated row is stored

var wg sync.WaitGroup

var concurrent = 1 // number of concurrent client performing search

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

// Write Benchmarking results to a series of json files
func WriteLog(benchmarkLog BenchmarkLog) (err error) {
	file, _ := json.MarshalIndent(benchmarkLog, "", " ")
	fmt.Println("Benchmark Log: \n")
	fmt.Println(benchmarkLog)
	err = ioutil.WriteFile(dir+"test_"+strconv.Itoa(benchmarkLog.EpochNum)+".json", file, 0644)
	return
}

// Connect to Milvus server, return MilvusClient object
func ConnectServer() (milvusClient milvus.MilvusClient, err error) {
	connectParam := milvus.ConnectParam{IPAddress: "localhost", Port: "19530"}
	ctx := context.TODO()
	milvusClient, err = milvus.NewMilvusClient(ctx, connectParam)
	if err != nil {
		log.Fatal("failed to connect to Milvus:", err.Error())
		return nil, err
	}
	log.Print("Connected to server")
	return milvusClient, nil
}

// Check whether or not collection with CollectionName exist
func CheckCollection(milvusClient milvus.MilvusClient) (hasColl bool, err error) {
	hasColl, _, err = milvusClient.HasCollection(
		context.Background(), // ctx
		collectionName,       // CollectionName
	)
	if err != nil {
		log.Fatal("failed to check whether collection exists:", err.Error())
		return hasColl, err
	}
	fmt.Printf("Milvus has Collection %s: %t \n", collectionName, hasColl)
	return hasColl, err
}

// List all collections available in Milvus
func ListCollections(milvusClient milvus.MilvusClient) {
	var collections []string
	collections, status, err := milvusClient.ListCollections(context.TODO())
	if err != nil {
		println("ShowCollections rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Show collections failed: " + status.GetMessage())
		return
	}
	println("ShowCollections: ")
	for i := 0; i < len(collections); i++ {
		println(" - " + collections[i])
	}
	return
}

// Create new Collection
func CreateCollection(milvusClient milvus.MilvusClient) (err error) {
	collectionParam := milvus.CollectionParam{
		collectionName,   // CollectionName
		512,              // Dimension of Vector
		1024,             // Index file size
		int64(milvus.IP), // Metric Type (Euclidean distance or Inner Product)
	}
	_, err = milvusClient.CreateCollection(context.Background(), collectionParam)
	if err != nil {
		log.Fatal("failed to create collection:", err.Error())
	}
	return
}

// Create Index for newly-inserted rows
func CreateIndex(milvusClient milvus.MilvusClient) (err error) {
	println("Start create index...")
	extraParams := "{\"nlist\" : 16384}"
	indexParam := milvus.IndexParam{
		collectionName, // Collection name
		milvus.IVFFLAT, // Type of Index
		extraParams,    // Additional parameter (Index type-specific)
	}
	status, err := milvusClient.CreateIndex(context.Background(), &indexParam)
	if err != nil {
		println("CreateIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("Create index success!")
	return
}

// Describe various details about the collection
func DescribeCollection(milvusClient milvus.MilvusClient) (err error) {
	//Describe index
	indexParam, status, err := milvusClient.GetIndexInfo(context.Background(), collectionName)
	if err != nil {
		println("DescribeIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Describe index failed: " + status.GetMessage())
	}
	println(indexParam.CollectionName + "----index type:" + strconv.Itoa(int(indexParam.IndexType)))

	collectionParam, status, err := milvusClient.GetCollectionInfo(context.Background(), collectionName)
	if err != nil {
		println("DescribeCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Create index failed: " + status.GetMessage())
		return
	}
	println("CollectionName:" + collectionParam.CollectionName + "----Dimension:" + strconv.Itoa(int(collectionParam.Dimension)) +
		"----IndexFileSize:" + strconv.Itoa(int(collectionParam.IndexFileSize)))

	//Number of rows in a Collection
	var collectionCount int64
	collectionCount, status, err = milvusClient.CountEntities(context.Background(), collectionName)
	if err != nil {
		println("CountCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Get collection count failed: " + status.GetMessage())
		return
	}
	println("Collection count: " + strconv.Itoa(int(collectionCount)))
	return
}

// Drop the Collection
func DropCollection(milvusClient milvus.MilvusClient) (err error) { //Drop index
	status, err := milvusClient.DropIndex(context.Background(), collectionName)
	if err != nil {
		println("DropIndex rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Drop index failed: " + status.GetMessage())
	}

	//Drop collection
	status, err = milvusClient.DropCollection(context.Background(), collectionName)
	hasCollection, status1, err := milvusClient.HasCollection(context.Background(), collectionName)
	if !status.Ok() || !status1.Ok() || hasCollection == true {
		println("Drop collection failed: " + status.GetMessage())
		return
	}
	println("Drop collection " + collectionName + " success!")
	return
}

// Insert n Random Row where n is equal to segment size
func InsertRandom(milvusClient milvus.MilvusClient, segment int, start int, end int) (err error) {
	pkIDs := make([]int64, 0, segment)
	embeddings := make([][]float32, 0, segment)

	// initiate random data
	for i := start; i < end; i++ {
		pkIDs = append(pkIDs, int64(i))
		v := make([]float32, 0, 512)
		for j := 0; j < 512; j++ {
			v = append(v, 1+rand.Float32()*(2-1))
		}
		embeddings = append(embeddings, v)
	}

	// Insert to collection
	rows := make([]milvus.Entity, segment)

	for i := 0; i < 10; i++ {
		rows[i].FloatData = embeddings[i]
	}
	insertParam := milvus.InsertParam{collectionName, "", rows, pkIDs}
	id_array, status, err := milvusClient.Insert(context.Background(), &insertParam)
	if err != nil {
		println("Insert rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Insert vector failed: " + status.GetMessage())
		return
	}
	if len(id_array) != int(10) {
		println("ERROR: return id array is null")
	}
	println("Insert vectors success!")
	return
}

// insert initial 10 rows from CSV to Collection
func LoadInsertFromCSV(milvusClient milvus.MilvusClient) (err error) {
	pkIDs := make([]int64, 0, initialRow)
	embeddings := make([][]float32, 0, initialRow)

	// load CSV and replace specific ID with loaded data
	f, err := os.Open(benchmarkDataset)
	if err != nil {
		log.Fatal("fail to load CSV file: ", err.Error())
		return
	}

	r := csv.NewReader(f)
	raw, err := r.ReadAll()
	if err != nil {
		log.Fatal("fail to read CSV file: ", err.Error())
		return
	}

	for idx, line := range raw {
		if idx > 0 {
			// ignore CSV header
			pkIDs = append(pkIDs, int64(idx-1))
			embedRaw := strings.Split(line[1], ",")
			embed := make([]float32, 0, 512)
			for _, el := range embedRaw {
				floatEl, _ := strconv.ParseFloat(el, 32)
				embed = append(embed, float32(floatEl))
			}
			embeddings = append(embeddings, embed)
		}
	}

	// Insert to collection
	rows := make([]milvus.Entity, 10)

	for i := 0; i < 10; i++ {
		rows[i].FloatData = embeddings[i]
	}
	insertParam := milvus.InsertParam{collectionName, "", rows, pkIDs}
	_, status, err := milvusClient.Insert(context.Background(), &insertParam)
	if err != nil {
		println("Insert rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Insert vector failed: " + status.GetMessage())
		return
	}
	return
}

func PopulateRandom(milvusClient milvus.MilvusClient, initial bool, offset int) {
	initialOffset := 0 // How many rows have been inserted before
	if initial == true {
		// if this is the first cluster of populate data, set the offset equal to the number of rows in pre-generated CSV
		initialOffset = initialRow
	}
	currentLoop := loop - (initialOffset / initialRow)
	for i := 0; i < currentLoop; i++ {
		fmt.Println("Inserting data at loop " + strconv.Itoa(i))
		start := i*segment + initialOffset + offset
		end := start + segment
		err := InsertRandom(milvusClient, segment, start, end)
		if err != nil {
			log.Fatal("failed to insert random data at loop "+strconv.Itoa(i+offset)+": ", err.Error())
		}
	}
	return
}

// Perform similarity search
func SimilaritySearch(milvusClient milvus.MilvusClient, epochNum int, indexName string, searchIndex int) (err error) {

	// Prepare Log
	var benchmarkLog BenchmarkLog
	benchmarkLog.Index = indexName
	benchmarkLog.EpochNum = epochNum
	benchmarkLog.Rows = segment * loop * cluster
	benchmarkLog.Timestamp = time.Now()
	benchmarkLog.MemoryUsage = 0 // TBA
	benchmarkLog.Concurrent = 1
	benchmarkLog.FloatBit = 32

	// Load CSV for search vector
	f, err := os.Open(benchmarkDataset)
	if err != nil {
		log.Fatal("fail to load CSV file: ", err.Error())
		return
	}
	r := csv.NewReader(f)
	raw, err := r.ReadAll()
	if err != nil {
		log.Fatal("fail to read CSV file: ", err.Error())
		return
	}

	// Create a vector to be searched from the CSV corresponding to certain index
	searchVector := make([]float32, 0, 512)
	for idx, line := range raw {
		if idx > 0 {
			if idx == searchIndex+1 {
				embedRaw := strings.Split(line[0], ",")
				for _, el := range embedRaw {
					floatEl, _ := strconv.ParseFloat(el, 32)
					searchVector = append(searchVector, float32(floatEl))
				}
			}
		}
	}

	//Construct vector to query
	searchRow := make([]milvus.Entity, 1)
	searchRow[0].FloatData = searchVector

	t0 := time.Now() // timestamp when collection is being loaded into memory
	loadCollectionParam := milvus.LoadCollectionParam{
		collectionName,
		nil, // partition tag list
	}
	status, err := milvusClient.LoadCollection(context.Background(), loadCollectionParam)
	if err != nil {
		println("PreloadCollection rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println(status.GetMessage())
	}
	t1 := time.Now()                   // timestamp when collection is loaded in memory
	benchmarkLog.LoadTime = t1.Sub(t0) // load time

	extraParams := "{\"nprobe\" : 32}"
	searchParam := milvus.SearchParam{collectionName, searchRow, 5, nil, extraParams}
	t2 := time.Now() // timestamp when search is initialized
	topkQueryResult, status, err := milvusClient.Search(context.Background(), searchParam)
	t3 := time.Now() // timestamp when search is over
	if err != nil {
		println("Search rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println("Search vectors failed: " + status.GetMessage())
	}
	benchmarkLog.Latency = t3.Sub(t2)   // search time
	benchmarkLog.TotalTime = t3.Sub(t0) // load + search time

	println("Search with index results: ")
	print(topkQueryResult.QueryResultList[0].Ids[0])
	print("        ")
	println(topkQueryResult.QueryResultList[0].Distances[0])

	err = WriteLog(benchmarkLog)
	if err != nil {
		log.Fatal("failed to write log:", err.Error())
		return
	}

	return
}

func runConcurrentRequests(milvusClient milvus.MilvusClient, lower int, upper int) {
	for i := lower; i < upper; i++ {
		err := SimilaritySearch(milvusClient, i, "IVF Flat", i%10)
		if err != nil {
			return
		}
	}
	wg.Done()
}

func main() {

	// Initialize flag options to run the program
	drop := flag.Bool("drop", false, "Drop the collection")
	init := flag.Bool("init", false, "Populate collection with random rows and initial rows from CSV")
	index := flag.Bool("index", false, "Index all newly-added rows")
	search := flag.Bool("search", false, "Perform concurrent search and record the log for each search")
	flag.Parse()

	// Connect the server
	fmt.Println("-----------------\n")
	log.Print("Connecting to server...")
	milvusClient, err := ConnectServer()
	if err != nil {
		return
	}

	// Check whether or not collection exists
	hasColl, err := CheckCollection(milvusClient)
	if err != nil {
		return
	}
	if !hasColl {
		// If collection does not exist, create it
		CreateCollection(milvusClient)
	}

	// Process "Drop" flag
	if *drop {
		_ = DropCollection(milvusClient)
		return
	}

	// Process "Init" flag
	if *init {
		err = LoadInsertFromCSV(milvusClient)
		if err != nil {
			return
		}

		for i := 0; i < cluster; i++ {
			initial := false
			if i == 0 {
				initial = true
			}
			PopulateRandom(milvusClient, initial, i*loop*segment)
		}
	}

	// Process "Index" flag
	if *index {
		future := async.Exec(func() interface{} {
			return CreateIndex(milvusClient)
		})
		future.Await()
	}

	// Process "Search" flag
	if *search {
		wg.Add(concurrent)
		for i := 0; i < concurrent; i++ {
			go runConcurrentRequests(milvusClient, i*searchEpoch, (i+1)*searchEpoch)
		}
		wg.Wait()
	}

	// Describe Collection details
	err = DescribeCollection(milvusClient)
	if err != nil {
		return
	}

	// Summarize benchmarking stats
	sortedLoad := stats.GetLoadArrSorted(searchEpoch, "./Benchmark/GPU_IVF_Flat/300K/")
	sortedLatency := stats.GetLatencyArrSorted(searchEpoch, "./Benchmark/GPU_IVF_Flat/300K/")
	stats.GetAvg(sortedLoad, "Load Time")
	stats.GetAvg(sortedLatency, "Latency")
	stats.GetPercentile(sortedLatency, 0.9, searchEpoch)
	stats.GetPercentile(sortedLatency, 0.1, searchEpoch)
	stats.GetPercentile(sortedLatency, 0.99, searchEpoch)
}
