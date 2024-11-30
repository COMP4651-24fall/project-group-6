# COMP4651-Second-hand Device Price Evaluation FaaS System

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jzfQvm5J)

A course project of HKUST COMP4651: Cloud Computing and Big Data Systems in 2024 Fall.

## Submission Info

Group No: 6
Group Members:

- HUANG, Yifu (20844438)
- HUANG, Ziyan (21161754)
- LI, Yifan (20945347)
- YIP, Valerie Fang Hong (21161340)

---

## Project Overview

This project focuses on **deployment and optimization of machine learning models in a serverless architecture** using **OpenFaaS, Kubernetes, and AWS EKS**. The main objectives include:

1. **Deploying a machine learning model** in a serverless architecture using OpenFaaS and Kubernetes on AWS EKS.
2. **Comparing serverless architecture with traditional server-based approaches** in terms of performance, cost, and scalability.
3. **Analyzing the trade-offs** between serverless and traditional architectures in terms of deployment and optimization of machine learning models.

## Developing and Testing the Function Locally

### Requirements

1. Install `faas-cli` ([installation guide](https://docs.openfaas.com/cli/install/)).
2. Pull the OpenFaaS Python template:
   ```bash
   faas-cli template store pull python3-http
   ```
   _(Note: This step uses a default Dockerfile as a workaround; future customizations may be made.)_

### Local Development

1. All development should be done in the `/src` folder, with `handler.py` as the entry point for the function.
2. To start local testing with OpenFaaS:
   ```bash
   faas-cli local-run -f project.yml
   ```
3. Test the function with the following `curl` command:
   ```bash
   curl http://127.0.0.1:8080 -d '[{
        'screen_size': 6.1,
        'rear_camera_mp': 12,
        'front_camera_mp': 12,
        'internal_memory': 128,
        'ram': 4,
        'battery': 3000,
        'weight': 175,
        'days_used': 30,
        'device_brand': 'OnePlus',
        'os': 'Android',
        '4g': 'yes',
        '5g': 'no'
    }]'
   ```

## Building and Deploying the Function

### Step 1: Creating the EKS Cluster (AWS EKS + Kubernetes)

1. Create an EKS cluster manually in the AWS Console with a node group (EC2 instances, tested with AMD).
   _(Note: Using `eksctl` was avoided due to limited IAM permissions.)_

### Step 2: Setting Up the Cluster on AWS EKS

1. Open [this guide](https://aws.amazon.com/blogs/opensource/deploy-openfaas-aws-eks/) to install OpenFaaS on the EKS cluster.
2. Configure AWS credentials:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name project
   ```
   Run `kubectl get nodes` to verify that the cluster is connected.

### Step 3: Building and Deploying the Serverless Function (OpenFaaS + Kubernetes on AWS EKS)

1. Log in to Docker:
   ```bash
   docker login
   ```
2. Set the Docker image prefix in `project.yml` to your Docker username.
3. Build and publish the Docker image:
   ```bash
   faas-cli publish -f project.yml --platforms "linux/amd64"
   ```
4. Deploy the function to AWS EKS:
   ```bash
   faas-cli deploy -f project.yml -g $OPENFAAS_URL
   ```

### Step 4: Testing the Deployment

Test the deployed function with:

```bash
curl $OPENFAAS_URL/function/cloud-project
```

## Building Machine Learning Models

### Model Files

The following files are required for the machine learning model:

1. `depreciation_model.h5` - Pretrained machine learning model.
2. `encoder.pkl` - Encoder for data preprocessing.
3. `scaler.pkl` - Scaler for data normalization.

### Train

To train the machine learning model and generate model checkpoints:

```bash
python ./src/train.py
```

### Inference

To test the inference locally using `handler.py`:

```bash
python ./src/handler.py
```

## Deploying on Traditional Server for Comparison

### Using Docker

1. **Set Up Server Environment**

   - Ensure Docker is installed on your server. Follow the [Docker installation guide](https://docs.docker.com/get-docker/) for your operating system.

2. **Build the Docker Image**

   - Navigate to the `/dockerfiles` directory and build the Docker image:
     ```bash
     docker build -t <your-docker-username>/<your-image-name> .
     ```

3. **Run the Docker Container**

   - Start the Docker container:
     ```bash
     docker run -p 5000:5000 <your-docker-username>/<your-image-name>
     ```

4. **Test the Traditional Deployment**
   - Send a request to the deployed service:
     ```bash
     curl http://127.0.0.1:5000/predict -d '{"quantity": 100}'
     ```

## Key Technologies

- **OpenFaaS**: A serverless framework for deploying functions on Kubernetes.
- **Kubernetes**: Container orchestration platform, used to manage the OpenFaaS deployment on AWS EKS.
- **AWS EKS**: Managed Kubernetes service by AWS, where OpenFaaS functions are deployed.
- **Docker**: Used for building and deploying containerized machine learning models both in the serverless and traditional environments.

---

## Deployment Guide

This document provides the steps for deploying the COMP4651 course project, which involves deploying a machine learning model using a serverless architecture (OpenFaaS + Kubernetes + AWS EKS) and comparing it to traditional server-based deployment.

### Prerequisites

Before starting, ensure you have the following tools installed:

- **faas-cli**: For interacting with OpenFaaS.  
  [Installation Guide](https://docs.openfaas.com/cli/install/)
- **Docker**: To build and run Docker containers.  
  [Installation Guide](https://docs.docker.com/get-docker/)
- **kubectl**: Kubernetes command-line tool for interacting with EKS.  
  [Installation Guide](https://kubernetes.io/docs/tasks/tools/install-kubectl/)

- **AWS CLI**: Command-line tool for interacting with AWS resources.  
  [Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

### Deployment Sequence

#### 1. **Set Up the Development Environment**

Ensure the necessary tools are installed (mentioned above).

#### 2. **Develop and Test the Function Locally**

- **Pull OpenFaaS Template**  
  First, pull the Python template for OpenFaaS:

  ```bash
  faas-cli template store pull python3-http
  ```

- **Local Development**  
  All development should take place in the `/src` directory, with `handler.py` as the entry point.
  Start the function locally using `faas-cli`:

  ```bash
  faas-cli local-run -f project.yml
  ```

  Test the function locally:

  ```bash
  curl http://127.0.0.1:8080 -d '[{
        'screen_size': 6.1,
        'rear_camera_mp': 12,
        'front_camera_mp': 12,
        'internal_memory': 128,
        'ram': 4,
        'battery': 3000,
        'weight': 175,
        'days_used': 30,
        'device_brand': 'OnePlus',
        'os': 'Android',
        '4g': 'yes',
        '5g': 'no'
    }]'
  ```

- **Ensure Local Functionality**  
  Make sure the function works as expected locally. This includes confirming that the correct model is loaded and can process the request properly.

#### 3. **Train the Machine Learning Models**

- **Train the Model**  
  To train the machine learning model and generate the necessary files (`depreciation_model.h5`, `encoder.pkl`, `scaler.pkl`), run the following command:

  ```bash
  python ./src/train.py
  ```

- **Test Inference**  
  Once the model is trained, test inference using the `handler.py` file:
  ```bash
  python ./src/handler.py
  ```
  Ensure that the model produces the expected output.

#### 4. **Set Up the AWS EKS Cluster**

- **Create EKS Cluster**  
  Manually create an EKS cluster through the AWS console. Since IAM role creation might require special permissions, you may need to do this manually.

- **Configure AWS CLI and kubectl**  
  Set up your AWS credentials and update the `kubectl` configuration to interact with the EKS cluster:

  ```bash
  aws eks update-kubeconfig --region us-east-1 --name project
  kubectl get nodes
  ```

  Ensure that you can retrieve the nodes of your EKS cluster.

#### 5. **Build and Deploy the Function to EKS**

- **Log in to Docker**  
  Make sure you are logged into Docker:

  ```bash
  docker login
  ```

- **Build Docker Image**  
  Using `faas-cli`, build and publish the Docker image for the function:

  ```bash
  faas-cli publish -f project.yml --platforms "linux/amd64"
  ```

- **Deploy the Function to EKS**  
  Deploy the function to your AWS EKS cluster:

  ```bash
  faas-cli deploy -f project.yml -g $OPENFAAS_URL
  ```

  This step will push the function to the OpenFaaS instance running on your Kubernetes cluster.

#### 6. **Test the Deployment**

- **Verify Function Deployment**  
  Once the function is deployed to the cloud, verify the deployment by sending a request:

  ```bash
  curl $OPENFAAS_URL/function/cloud-project
  ```

  If successful, you should receive a response from your function indicating that it has been deployed and is running correctly.

#### 7. **Deploy on Traditional Server for Comparison**

To compare serverless and traditional deployments:

- **Set Up Traditional Server**  
  On your traditional server, install Docker and build the same Docker image for the function:

  ```bash
  cd dockerfiles
  docker build -t <your-docker-username>/<your-image-name>:<tag>
  ```

- **Run the Docker Container**  
  Run the container on the server, mapping the appropriate port (e.g., port 5000):

  ```bash
  docker run -d -p 5000:5000 <your-docker-username>/<your-image-name>:<tag>
  ```

  This will allow you to test the function in a traditional server environment for comparison.

#### 8. **Analyze and Compare**

- After deploying both on the OpenFaaS serverless environment and a traditional server, analyze the performance, scalability, and cost of both architectures. Key aspects to compare include:
  - **Performance**: Response times and throughput under various loads.
  - **Scalability**: Discussing how each architecture handles scaling under high loads.

##### 8.1 Performance Test

- Usage
  Install dependencies:

  ```bash
  pip install -r ./tests/requirements.txt
  ```

- Run tests:

  ```bash
  python -m tests.performance.deployment_performance_test
  ```

Test Parameters:

- requests_per_round: Number of requests per test round
- max_workers: Maximum concurrent requests
- test_rounds: Number of test rounds
- timeout: Request timeout (seconds)

### Conclusion

Following these steps will enable you to successfully deploy your machine learning model using OpenFaaS and Kubernetes on AWS EKS, as well as compare it to a traditional server-based deployment for performance, scalability, and cost analysis.
