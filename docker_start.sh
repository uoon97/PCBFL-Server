# Install Docker
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -a -G docker $USER
newgrp docker