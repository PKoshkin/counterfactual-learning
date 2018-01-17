XGBFLAGS=-I xgboost/include -I xgboost/rabit/include -fopenmp
XGBLIBS=xgboost/lib/libxgboost.a xgboost/rabit/lib/librabit.a xgboost/dmlc-core/libdmlc.a
CFLAGS=-std=c++11

all: main

main: learning utils
	g++ main.cpp learning.o utils.o -o main $(CFLAGS) $(XGBFLAGS) $(XGBLIBS)

learning:
	g++ -c learning.cpp -o learning.o $(CFLAGS) $(XGBFLAGS)

utils:
	g++ -c utils.cpp -o utils.o $(CFLAGS)

clean:
	rm main *.o
