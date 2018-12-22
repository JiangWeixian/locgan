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

```bash
sudo docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon
# check datasets config in dataloader/config.py
python train_mask_net.py
# see train loss detain in localhost:8888
```

## TODO

* [x] - dataloader
* [x] - model file
* [x] - train it

## Changelog

* init
* rm unusecode
  > reduce network only to G/D
* complete dataloader

**Warning**

* test&eval files not valid