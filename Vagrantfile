VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(2) do |config|
  
    config.vm.box = "ubuntu/trusty64"
    config.vm.hostname = "vagrant-docker-anaconda"

    config.vm.network "private_network", ip: "192.168.50.100"
    config.vm.network "forwarded_port", guest: 80, host: 8080
    config.vm.provider :virtualbox do |vb|
        vb.customize ["modifyvm", :id, "--memory", "1024", "--cpus", "2"]
    end

    config.vm.synced_folder ".", "/vagrant", :type => "nfs"

    # IPython Notebook
    config.vm.network :forwarded_port, host: 8888, guest: 8888
    # Flask
    config.vm.network :forwarded_port, host: 5000, guest: 5000
    # MongoDB
    config.vm.network :forwarded_port, host: 27017, guest: 27017
    config.vm.network :forwarded_port, host: 27018, guest: 27018
    # Postgres
    config.vm.network :forwarded_port, host: 5432, guest: 5432

    config.vm.provision "shell", path: "setup.sh"
    config.vm.provision :docker
    config.vm.provision :docker_compose, yml: "/vagrant/docker-compose.yml", rebuild: true, run: "always"

end