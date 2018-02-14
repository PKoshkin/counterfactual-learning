XGBFLAGS=-I xgboost/include -I xgboost/rabit/include -fopenmp
XGBLIBS=xgboost/lib/libxgboost.a xgboost/rabit/lib/librabit.a xgboost/dmlc-core/libdmlc.a
CFLAGS=-std=c++11

all: al_main

al_main: metric model utils pool counterfactural_model active_learning_strategies active_learning_algo al_utils xgboost_model
	g++ al_main.cpp counterfactural_model.o metric.o utils.o model.o active_learning_algo.o pool.o active_learning_strategies.o al_utils.o xgboost_model.o -o main $(CFLAGS) $(XGBFLAGS) $(XGBLIBS)

main: metric model utils pool counterfactural_model xgboost_model
	g++ main.cpp counterfactural_model.o metric.o utils.o model.o pool.o xgboost_model.o -o main $(CFLAGS) $(XGBFLAGS) $(XGBLIBS)

al_utils:
	g++ -c al_utils.cpp -o al_utils.o $(CFLAGS)

counterfactural_model:
	g++ -c counterfactural_model.cpp -o counterfactural_model.o $(CFLAGS)

active_learning_strategies:
	g++ -c active_learning_strategies.cpp -o active_learning_strategies.o $(CFLAGS)

active_learning_algo:
	g++ -c active_learning_algo.cpp -o active_learning_algo.o $(CFLAGS)

pool:
	g++ -c pool.cpp -o pool.o $(CFLAGS)

model:
	g++ -c model.cpp -o model.o $(CFLAGS)

metric:
	g++ -c metric.cpp -o metric.o $(CFLAGS)

utils:
	g++ -c utils.cpp -o utils.o $(CFLAGS)

xgboost_model:
	g++ -c xgboost_model.cpp -o xgboost_model.o $(CFLAGS) $(XGBFLAGS)

clean:
	rm main *.o
