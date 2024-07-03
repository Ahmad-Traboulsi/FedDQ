# FedDQ: Federated Data Quality with Federated Learning

This repository contains the implementation of a federated learning simulation for the Federated Data Quality (FedDQ) project. The goal of FedDQ is to improve data quality in federated learning settings.

## Table of Contents

- [Introduction](#introduction)
- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository provides the necessary code to simulate the FedDQ approach. It includes components for clients, a federated server, model operations using PyTorch, dataset loading, and knowledge graph pre-processing.

## Components

The repository consists of the following main components:

- `client.py`: Contains the code for the federated learning clients. Each client participates in the training process by contributing its local model updates.
- `server.py`: Implements the federated server responsible for aggregating the model updates from the clients and updating the global model.
- `model.py`: Defines the model architecture and operations using PyTorch. This includes training, evaluation, and inference functions.
- `dataset.py`: Provides functionality for loading and preprocessing the dataset used in the federated learning simulation.
- `utils.py`: Contains utility functions for knowledge graph pre-processing and other auxiliary operations.
- `main.py`: Contains the main driver for simulation execution

## Installation

To use this project, please follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/Ahmad-Traboulsi/FedDQ.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

   Note: Make sure you have Python and pip installed on your system.

## Usage

To run the FedDQ simulation, follow these steps:

1. Go to the project directory:

   ```shell
   cd FedDQ
   ```

2. Customize the configurations and settings in the code files according to your requirements.

3. Execute the main file to launch server and clients:

   ```shell
   python main.py
   ```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Let's collaborate to enhance the FedDQ project together!

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as per the license terms.
