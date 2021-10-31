#DeepSearchers 

We use TensorRT for Yolov5 infer

## Download weights

- Download weights from this [link](https://croc.disk.croc.ru/public/h8gGhSOI)
- Unzip weights in project's folder

Your project should look like this:

    - static
    - templates
    - weights
    app.py
    Dockerfile
    ...


## Buid and Run Docker


```bash
    docker build -t evraz-ai .
    docker run --gpus all evraz-ai:latest 
```

When docker starts follow to your host. 

## Thanks to

- [Yolov5](https://github.com/ultralytics/yolov5)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
