.PHONY: all clean
CC = gcc
CFLAGS = -std=c99 -O2 -w
LDFLAGS+=-lm     # link to math library
TARGET = train test
ITERATION = 900
RM = rm -f
MODELS = model_0?.txt
RESULTS = result?.txt

# all: $(TARGET) $(run) $(clean)
all: $(TARGET)
# type make/make all to compile test_hmm

train: train.c
	$(CC) $(CFLAGS) $< -o $@

test: test.c
	$(CC) $(CFLAGS) $< -o $@

# accuracy: accuracy.c
# 	$(CC) $(CFLAGS) $< -o $@

run: run_train run_test

run_train: train
	./train $(ITERATION) model_init.txt seq_model_01.txt model_01.txt
	./train $(ITERATION) model_init.txt seq_model_02.txt model_02.txt
	./train $(ITERATION) model_init.txt seq_model_03.txt model_03.txt
	./train $(ITERATION) model_init.txt seq_model_04.txt model_04.txt
	./train $(ITERATION) model_init.txt seq_model_05.txt model_05.txt

run_test: test
	./test modellist.txt testing_data1.txt result1.txt
	./test modellist.txt testing_data2.txt result2.txt
# ./accuracy

clean:
	# type make clean to remove the compiled file
	$(RM) $(TARGET)