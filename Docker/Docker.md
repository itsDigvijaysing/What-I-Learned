- It's Virtualization software, by packaging application with all the necessary dependencies, config, system tools & runtimes in Docker Artifact.

> ![Docker Cycle](Docker%20Cycle.png)
- Reasons to use 
	1. Different Developers working with Different OS.
	2. Can solve the problem of versions of Software's.
	3. Can fix the issue of manually doing setup for development.
- It has it's own isolated environment & Postgres packaged with all dependencies and configs.
- Can run different version of application on same project simultaneously.
- It use Hypervisor Layer with lightweight Linux Distro & that way it can run on Windows as well, previously, It was only made for Linux distro (Docker Desktop).

> ![Docker Image vs Docker Container](Docker%20Image%20Container.png)

- From One Image we can run multiple Containers.
- Docker Registries, a storage and distribution system for Docker Images.
- They have Software Version as well, but in Docker Image we call them as Tags (version).
- We can pull Docker Image using Docker CLI by entering the command.
- [Use this Docker commands](https://docs.docker.com/engine/reference/commandline/cli/)

> ![Docker Connection](Docker%20Connection.png)