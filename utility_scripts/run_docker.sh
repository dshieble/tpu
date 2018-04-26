# git pull; ./utility_scripts/run_docker.sh
# Build the image and label it as tpu-example on the Docker registry on p3
nvidia-docker build -t serrep1node.services.brown.edu:5000/tpu .

#Run the container
nvidia-docker run -d --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs/cluster_projects/tpu serrep$node.services.brown.edu:5000/tpu bash start_gpu_worker.sh 0
