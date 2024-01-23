# Neural Network Classifier for Iris Dataset
Welcome to my project, a neural network-based classifier designed to work with the famous Iris dataset. This project uses PyTorch, , to create and train a neural network capable of classifying Iris flowers into different species. 

### Key Features
<br>
 - Neural Network Architecture: the neural network architecture designed for the Iris dataset classification task.
 - Training Process: steps involved in training the model, including data preprocessing, model configuration, and optimization.
 - Model Evaluation: evaluating the model's performance and making predictions on new data.


## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_inference_data.csv
│   └── iris_train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   ├── various model files
│   └── scaler files for scaling the inference data
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── training.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
├── README.md
└── requirements.txt          # All libraries and their versions necessary to run the project
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project like:
 - the path to certain resource files,
 - constant values, 
 - hyperparameters for an ML model. 

## Data:
For generating the data, we use the script located at `data_process/data_generation.py`. It downloads iris dataset from the Wikipedia page. The generated data is used to train the model and to test the inference. 

## Training:
The training phase of the ML pipeline includes
 - preprocessing of data, 
 - training of the models with different parameters
 - choosing best performing model. 
All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -dit training_image
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.

