
# Remove all unused containers, networks, images, and build cache
docker system prune -a --volumes -f
# Remove build cache
docker builder prune -a -f

# Remove all unused images
docker image prune -a -f

# Remove all stopped containers
docker container prune -f

# Remove all unused volumes
docker volume prune -f

# Remove all unused networks
docker network prune -f

# Remove all dangling images
docker image prune -f

# Stop all running containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Remove all images
docker rmi $(docker images -q) -f

# Remove all volumes
docker volume rm $(docker volume ls -q)

# Remove all networks (except default ones)
docker network rm $(docker network ls -q)

# Clean up build cache
docker builder prune -a -f

docker build -t river-lambda .

docker tag river-lambda:latest 141706873475.dkr.ecr.us-east-1.amazonaws.com/river-lambda:latest
docker push 141706873475.dkr.ecr.us-east-1.amazonaws.com/river-lambda:latest

aws lambda update-function-code \
    --function-name river-task-assignment \
    --image-uri 141706873475.dkr.ecr.us-east-1.amazonaws.com/river-lambda:latest \
    --region us-east-1

