# LOCGAN
> init

## DEP

* docker - using torrvision/crayon for tensorboard

  ```bash
  sudo apt-get install docker.io
  ```

  then follow docs in [torchvision/crayon](https://github.com/torrvision/crayon)

  **maybe you need vpn to download docker and docker image**

* pytorchv0.3 - sorry for old version
* gpu mode - **only in cuda**

## Run

```
sudo docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon
```

## TODO

* [x] - dataloader
* [x] - model file
* [ ] - train it

## Changelog

* init
* rm unusecode
  > reduce network only to G/D
* complete dataloader

**Warning**

* test&eval files not valid