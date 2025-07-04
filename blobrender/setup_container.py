import os 
import subprocess
import argparse
import yaml
from blobrender.paths import CONTAINERS, CONFIGS
from blobrender.tools import update_yaml

def check_singularity_remote_login():
    try:
        result = subprocess.run(
            ["singularity", "remote", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Look for login confirmation in output
        output = result.stdout + result.stderr
        if (
            "Logged in as:" in output
            or "Valid authentication token set" in output
            or "Access Token Verified" in output
        ):
            return True
    except Exception as e:
        print("Could not check Singularity remote login status. Please ensure Singularity is installed.")
        print(e)
        return False
    
    
def singularity_setup(use_remote=False, use_local_file=False):
    if use_local_file:
        sif_name = "blobrender-local.sif"
    else:
        sif_name = "wsclean-1.6.3-dockerhub.sif"
    wsclean_sif = os.path.join(CONTAINERS, sif_name)
    local_def = os.path.join(CONTAINERS, "blobrender-singularity.def")

    if os.path.exists(wsclean_sif):
        print(f"Singularity container already exists at {wsclean_sif}. Skipping build/pull.")
    else:
        os.makedirs(CONTAINERS, exist_ok=True)
        if use_local_file:
            if use_remote:
                if not check_singularity_remote_login():
                    raise RuntimeError("Remote build requested but not logged in. Please run: singularity remote login")
                print("Using remote builder to build Singularity container from local definition file.")
                build_cmd = [
                    "singularity", "build", "--remote", wsclean_sif, local_def
                ]
            else:
                print(f"Building Singularity container from local definition file: {local_def}")
                build_cmd = [
                    "singularity", "build", wsclean_sif, local_def
                ]
            subprocess.run(build_cmd, check=True)
        else:
            if use_remote:
                if not check_singularity_remote_login():
                    raise RuntimeError("Remote build requested but not logged in. Please run: singularity remote login")
                print("Using remote builder to build Singularity container from DockerHub image.")
                build_cmd = [
                    "singularity", "build", "--remote", wsclean_sif, "docker://stimela/wsclean:1.6.3"
                ]
                subprocess.run(build_cmd, check=True)
            else:
                print("Pulling Singularity container from DockerHub locally.")
                pull_cmd = [
                    "singularity", "pull", f"--dir={CONTAINERS}", "docker://stimela/wsclean:1.6.3"
                ]
                subprocess.run(pull_cmd, check=True)
            print("Singularity container downloaded.")
    return sif_name

def docker_image_exists(image_name):
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return bool(result.stdout.strip())

def docker_setup(use_local_file=False):
    wsclean_docker = "stimela/wsclean:1.6.3"
    local_dockerfile = os.path.join(CONTAINERS, "blobrender-dockerfile")
    local_image_name = "blobrender-docker"
    if use_local_file:
        if docker_image_exists(local_image_name):
            print(f"Docker image '{local_image_name}' already exists. Skipping build.")
        else:
            print(f"Building Docker image from local Dockerfile: {local_dockerfile}")
            build_cmd = [
                "docker", "build", "-f", local_dockerfile, "-t", local_image_name, CONTAINERS
            ]
            subprocess.run(build_cmd, check=True)
        imagename = local_image_name
    else:
        if docker_image_exists(wsclean_docker):
            print(f"Docker image '{wsclean_docker}' already exists. Skipping pull.")
        else:
            print("Pulling Docker image from Docker Hub.")
            pull_cmd = ["docker", "pull", wsclean_docker]
            subprocess.run(pull_cmd, check=True)
        imagename = wsclean_docker
    return imagename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filetype",
        choices=["docker", "singularity"],
        required=False,
        help="Choose either 'docker' or 'singularity' (required unless --nocontainer is set)"
    )
    parser.add_argument(
        "--nocontainer",
        action="store_true",
        help="If set, do not use a container and ignore --filetype. This assumes you have wsclean installed locally."
    )
    parser.add_argument(
        "--local-file",
        action="store_true",
        help="If set, build the container from the local file in the containers folder instead of pulling/building from DockerHub."
    )
    parser.add_argument(
        "--remote-build",
        action="store_true",
        help="If set, use remote build for Singularity (ignored for Docker)."
    )

    args = parser.parse_args()

    # Make filetype case-insensitive
    if args.filetype:
        filetype = args.filetype.lower()
    else:
        filetype = None
    
    
    if args.nocontainer:
        print("You chose not to use a container.")
        container_name = 'None'
        args.filetype = 'None'
    else: # If a container is to be used
        if not args.filetype:
            parser.error("--filetype is required unless --nocontainer is set")
        print(f"You chose to use a container of type: {args.filetype}")
        
        if args.filetype == "singularity":
            container_name = singularity_setup(use_remote=args.remote_build, use_local_file=args.local_file)
        elif filetype == "docker":
            if args.remote_build:
                raise ValueError("Cannot use remote build with Docker. Option only available for Singularity.")
            container_name = docker_setup(use_local_file=args.local_file)
        else:
            parser.error("Invalid --filetype. Choose either 'docker' or 'singularity'.")

    yaml_path = os.path.join(CONFIGS, "default_prediction.yaml")
    update_yaml("container_name", container_name, yaml_path)
    update_yaml("container_type", args.filetype, yaml_path)



if __name__ == "__main__":
    main()
