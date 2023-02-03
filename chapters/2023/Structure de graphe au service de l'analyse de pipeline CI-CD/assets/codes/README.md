# GitHub action analyser
e
## Development
```shell
python -m pip install -r requirements.txt
```

## Build Docker image
```shell
docker build -t github_action_parser:latest .
```

## Execution
```shell
docker-compose up
# Or
./run.sh config_file.yml
```

By default, the application follows the behaviour described in [config/run-config.yaml](./config/run-config.yaml). You 
can change the behaviour by updating this file or create a new one that follows the same structure and change the
[docker image entrypoint](./Dockerfile) argument by your filename.

```shell
python3 main.py config/run-config.yaml
```

If you get trouble during image generation, you can regenerate the images using this:
```shell
dot -Tpng ./images/file.dot >./images/file.png
```
