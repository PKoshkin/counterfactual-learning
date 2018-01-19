XGBFLAGS=-I xgboost/include -I xgboost/rabit/include -fopenmp
XGBLIBS=xgboost/lib/libxgboost.a xgboost/rabit/lib/librabit.a xgboost/dmlc-core/libdmlc.a
CFLAGS=-std=c++11

all: main

main: metric model utils
	g++ main.cpp counterfactural_model.h metric.o utils.o model.o -o main $(CFLAGS)

model:
	g++ -c model.cpp -o model.o $(CFLAGS)

metric:
	g++ -c metric.cpp -o metric.o $(CFLAGS)

utils:
	g++ -c utils.cpp -o utils.o $(CFLAGS)

clean:
	rm main *.o
