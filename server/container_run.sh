docker build -t fl_server:latest .
docker network create mynet

docker run -d -p 27017:27017 --network mynet -e MONGODB_INITDB_ROOT_USERNAME=root -e MONGODB_INITDB_ROOT_PASSWORD=password -e MONGODB_BIND_ADDRESS=0.0.0.0 mongo:6.0
docker run -d --privileged -p 5000:5000 --network mynet fl_server:latest