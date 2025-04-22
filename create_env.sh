##################################################################
#################### Configuration Parameters ####################
##################################################################


# The configuration parameters listed below are utilized for creating a Determined AI container, which functions as a virtual machine equipped with GPUs.

# GPU_NUM: Defines the number of GPU units to be utilized.
# Example value: "2" signifies that four GPUs are to be used.
GPU_NUM="4"

# GPU_TYPE: Specifies the model of the GPU to be used.
# Example value: "128c256t_768_4090" indicates that NVIDIA GeForce RTX 4090 is the GPU model on LINs Lab cluster, other resources are: "64c128t_512_3090"; "128c256t_768_4090"; "64c128t_512_4090"; "56c112t_768_L40"; "temp".
GPU_TYPE="128c256t_768_4090"
# GPU_TYPE="64c128t_512_4090"
# GPU_TYPE="128c256t_768_6000Ada"
# GPU_TYPE="56c112t_768_L40"
# GPU_TYPE="64c128t_512_3090"

# Calculate the amount of memory. Each GPU corresponds to 4G of memory.
MEMORY=$(($GPU_NUM * 16))G

# IMAGE: Denotes the Docker image to be used for creating the container environment.
# Example value: "harbor.lins.lab/library/public_vision:v1.1" refers to a specific version of an image stored in a Docker registry.
IMAGE="harbor.lins.lab/library/public_vision:v1.1"

# ROOT_PATH: Determine the absolute path of the project, which contains the code, data, etc.
# Example value: "pwd" shows the current project directory's absolute path, works in almost all scenarios.
ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DATA_PATH: Defines the file system location for additional data beyond what is already in the project directory.
# Example value: "/labdata0/linslab" indicates the directory path for storing public data.
DATA_PATH="/labdata1/yrk/md"

##################################################################
################# End of Configuration Parameters ################
##################################################################

# Create a temporary file in the system's temporary directory.
TEMP_FILE=$(mktemp /tmp/temp_env.XXXXXX.yaml)

# Populate the temporary 'env.yaml' file with the configuration details.
cat > "$TEMP_FILE" << EOF
# Description section: specifies the root path of the project or script.
description: $(basename "$(dirname "$ROOT_PATH")")/$(basename "$ROOT_PATH")

# Resources section: defines the hardware resources to be allocated.
resources:
    slots: $GPU_NUM  # Number of GPU slots to be allocated.
    resource_pool: $GPU_TYPE  # Resource pool specification, dynamically incorporating the GPU type.
    shm_size: $MEMORY  # Shared memory size, using the value from the MEMORY parameter.

# Bind mounts section: maps host paths to container paths for data access and script execution.
bind_mounts:
    - host_path: $ROOT_PATH  # Map the project directory to a path inside the container.
      container_path: /run/determined/workdir/home
    - host_path: $DATA_PATH  # Map the additional data storage path to a path inside the container.
      container_path: /run/determined/workdir/home/md

# Environment section: specifies the container image to be used.
environment:
    image: $IMAGE  # Docker image to be used for the environment.

EOF

# Execute the Determined AI command to start a shell with the specified configuration file.
det shell start --config-file "$TEMP_FILE"