import os
import yaml
import random
import string
from prefect import task, flow
from prefect.blocks.system import String
import requests
import base64

@task
def template_job_yaml(job_name, namespace, docker_image, worker_replicas, gpu_count, gpu_vram, gpu_type=None):
    yaml_template = '''
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  generateName: {job_name}
  namespace: {namespace}
spec:
  cleanPodPolicy: OnSuccess
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: pytorch
              image: {docker_image}
    Worker:
      replicas: {worker_replicas}
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: pytorch
              image: {docker_image}
              resources:
                limits:
                  nvidia.com/gpu: {gpu_count}
                  nvidia.com/gpu-vram: {gpu_vram}
          {node_selector}
'''
    node_selector = ''
    if gpu_type:
        node_selector = f'''
          nodeSelector:
            gpu-type: {gpu_type}
'''
    yaml_content = yaml_template.format(
        job_name=job_name,
        namespace=namespace,
        docker_image=docker_image,
        worker_replicas=worker_replicas,
        gpu_count=gpu_count,
        gpu_vram=gpu_vram,
        node_selector=node_selector
    )
    return yaml.safe_load(yaml_content)

@task
def add_to_gitea_repo(gitea_url, gitea_token, repo_path, job_name, yaml_data, commit_message):
    file_path = f"{job_name}.yaml"
    yaml_string = yaml.dump(yaml_data)
    encoded_content = base64.b64encode(yaml_string.encode()).decode()

    branch = "main"
    url = f"{gitea_url}/api/v1/repos/{repo_path}/contents/{file_path}?ref={branch}"
    headers = {
        "Authorization": f"token {gitea_token}",
        "Content-Type": "application/json"
    }

    data = {
        "content": encoded_content,
        "message": commit_message
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"File '{file_path}' added to Gitea repository successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to add file '{file_path}' to Gitea repository. Error: {str(e)}")
        print("Response:", response.text)

# Load Prefect blocks
gitea_url_block = String.load("gitea-url")
gitea_token_block = String.load("gitea-token")

@flow(name="Training Pipeline")
def training_pipeline(
    job_name: str,
    namespace: str,
    docker_image: str,
    worker_replicas: int,
    gpu_count: int,
    gpu_vram: str,
    gpu_type: str = None,
):
    yaml_data = template_job_yaml(
        job_name,
        namespace,
        docker_image,
        worker_replicas,
        gpu_count,
        gpu_vram,
        gpu_type,
    )

    gitea_url = gitea_url_block.value
    gitea_token = gitea_token_block.value
    repo_path = "gitea_admin/pebble-fleet-training"
    commit_message = "Add PyTorch job YAML"

    add_to_gitea_repo(
        gitea_url,
        gitea_token,
        repo_path,
        job_name,
        yaml_data,
        commit_message,
    )

if __name__ == "__main__":
    # Run the flow
    training_pipeline(
        job_name="pytorch-job",
        namespace="pebble",
        docker_image="orl888/pebble-training:v1",
        worker_replicas=2,
        gpu_count=1,
        gpu_vram="8G",
        gpu_type="A100",
    )