<!-- vChat/DEPLOY_DSMLP.md -->
# Guide: Deploying vChat on UCSD DSMLP Kubernetes

This guide provides step-by-step instructions for deploying the vChat application on the UCSD DSMLP Kubernetes cluster. This process involves building a Docker image on your local computer, pushing it to a public registry, and then using `kubectl` on the DSMLP login node to deploy it.

## Prerequisites

1.  **SSH Access**: You must have SSH access to `dsmlp-login.ucsd.edu`.
2.  **Docker Desktop**: Docker must be installed and running on your local computer.
3.  **Docker Hub Account**: You need an account on [Docker Hub](https://hub.docker.com/) (or another container registry accessible from DSMLP). This is where you'll store your application's image.
4.  **`kubectl`**: The Kubernetes command-line tool. You will use this on the DSMLP login node, where it is pre-configured.

---

## Step 1: Build and Push the Docker Image

This step is done **on your local computer**. The goal is to package the application into a Docker image and upload it to Docker Hub so the DSMLP cluster can download it.

1.  **Open a terminal** and navigate to the `vChat` directory of your project.

2.  **Log in to Docker Hub**:
    ```bash
    docker login
    ```
    Enter your Docker Hub username and password when prompted.

3.  **Build the image**: Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username. The `.` at the end is important—it tells Docker to use the current directory as the build context.
    ```bash
    docker build -t YOUR_DOCKERHUB_USERNAME/vchat:latest .
    ```
    This command can take several minutes the first time it's run.

4.  **Push the image to Docker Hub**: Again, replace `YOUR_DOCKERHUB_USERNAME`.
    ```bash
    docker push YOUR_DOCKERHUB_USERNAME/vchat:latest
    ```
    After this step, your application image is now publicly available for the cluster to pull.

---

## Step 2: Prepare and Deploy on DSMLP

Now, you will connect to the DSMLP cluster and use the provided Kubernetes manifests to run your application.

1.  **SSH into the DSMLP login node**:
    ```bash
    ssh your_username@dsmlp-login.ucsd.edu
    ```

2.  **Clone your project repository**: You need the `k8s` manifest files on the login node.
    ```bash
    git clone <your-repo-url>
    cd vChat/
    ```

3.  **Edit the Deployment File**: Before you can deploy, you must tell Kubernetes which Docker image to use.
    *   Open `k8s/deployment.yaml` with a terminal editor like `nano` or `vim`:
        ```bash
        nano k8s/deployment.yaml
        ```
    *   Find the line that says `image: docker.io/YOUR_DOCKERHUB_USERNAME/vchat:latest`.
    *   Change `YOUR_DOCKERHUB_USERNAME` to your actual Docker Hub username.
    *   Save and exit the editor (in `nano`, press `Ctrl+X`, then `Y`, then `Enter`).

4.  **Apply the Kubernetes Manifests**: These commands tell Kubernetes to create the resources needed for your application.
    *   **Create the Persistent Volume Claim (PVC)** for storage:
        ```bash
        kubectl apply -f k8s/pvc.yaml
        ```
    *   **Create the Deployment** to run your app pod:
        ```bash
        kubectl apply -f k8s/deployment.yaml
        ```
    *   **Create the Service** to expose your app to the network:
        ```bash
        kubectl apply -f k8s/service.yaml
        ```

---

## Step 3: Check Status and Access Your Application

1.  **Check the Pod Status**: Your application is running in a "pod". Check if it's starting up correctly.
    ```bash
    kubectl get pods -w
    ```
    Wait for the STATUS to change from `ContainerCreating` to `Running`. This may take a few minutes as the cluster pulls your Docker image and downloads the large ML models for the first time. If it gets stuck or shows an `Error` or `ImagePullBackOff` status, proceed to the next step for debugging.

2.  **View Logs (Important for Debugging)**: To see the output from your application, including model download progress or any errors:
    ```bash
    # First, get your pod's exact name
    kubectl get pods

    # Then, use that name to view the logs
    kubectl logs -f <your-pod-name-from-above>
    ```

3.  **Find Your Service's Port**: To access the web UI, you need to find the port Kubernetes assigned.
    ```bash
    kubectl get service vchat-service
    ```
    Look at the `PORT(S)` column. You will see something like `8000:3XXXX/TCP`. The `3XXXX` number is your `NodePort`.

4.  **Find Your Node's Address**:
    ```bash
    kubectl get pod <your-pod-name> -o wide
    ```
    Look at the `NODE` column. It will give you the name of the node your pod is running on, for example, `dsmlp-gpu01.ucsd.edu`.

5.  **Access the Web UI**: Combine the node name and the `NodePort` in your browser:
    *   **URL Format**: `http://<node-name>:<node-port>`
    *   **Example**: `http://dsmlp-gpu01.ucsd.edu:31234`

You should now be able to see and interact with your vChat application running on the DSMLP cluster!

---

## Step 4: Cleaning Up

When you are finished, it's important to delete your Kubernetes resources to free up the GPU and storage for others.

Run these commands from the `vChat` directory on the DSMLP login node:
```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml

# WARNING: This command will delete the persistent volume claim and ALL data
# stored in it (cached models, downloaded videos, etc.).
kubectl delete -f k8s/pvc.yaml