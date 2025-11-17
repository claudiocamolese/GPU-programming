all:
	nvcc -rdc=true train_mnist.cu src/cnn_layers.cu src/common.cu src/data_loader.cu src/preprocessing.cu src/kernel_functions.cu -lm -o train_model

test:
	nvcc -rdc=true tests/run_training_tests.cu src/cnn_layers.cu src/common.cu src/data_loader.cu src/preprocessing.cu src/kernel_functions.cu tests/test_utils.cu tests/test_data_prep.cu -lm -o test
