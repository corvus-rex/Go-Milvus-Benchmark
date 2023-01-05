# Go Milvus Benchmark
 A simple program to test the capability of Milvus as a Vector Similarity Search library


This version utilizes a GPU-enabled Milvus at version 1.0.1

# Installation
Follow the documentation in order to install and setup a GPU-enabled Milvus in your localhost 
https://milvus.io/docs/v1.0.0/milvus_docker-gpu.md 

# Overview
The program will do the following based on flags user put before runtime
1) Connect to Milvus server
2) Create a new collection based on the `collectionName` global variable in the program if the collection does not exist
3) Populate the first n rows based on the pre-generated data from `benchmark_dataset.csv`. This data will serve as search vectors during the similarity search process
4) Populate the remaining rows with random 512-dimension vector. The number of rows including the pre-generated ones totalled to 300,000 entities. Number of rows is determined by the product of `loop`, `segment` and `cluster` global variables within the program. This is done in order to avoid max gRPC message limit.
5) Index all newly-added rows
6) Perform concurrent similarity search requests. Program will create a number of concurrent Goroutines determined by the `concurrent` global variable and each Goroutines will perform a number of search requests determined by `searchEpoch`global variable. The program will perform in total a number of requests equal to `concurrent` and `searchEpoch`
7) Logs all the search requests in `./Benchmark/GPU_IVF_Flat/300k/` directory by default as json file corresponding to each search request
8) Parse all json files and benchmark result to calculate and print latency statistics

# Usage
In order to run the desired functionality within the program, you need to supply it with the appropriate flags before runtime

- `-drop`: Boolean value, if set to true will drop the collection corresponding to `collectionName` global variable. Set to false by default. Will ignore other flags if set to true.
- `-init`: Boolean value, if set to true will add pre-generated rows and random rows to the collection. Set to false by default.
- `-search`: Boolean value, if set to true will perform concurrent search and log the result to json files. Set to false by default.
- `-index`: Boolean value, if set to true will index all newly-added rows. Set to false by default.

Example usage:
```go run . -init=True -index=True -search=True```
 will populate the collection with 300,000 rows (by default), index the rows, perform search and log them.
```go run . -drop=True```
 will drop the collection