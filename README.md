# traffic-lights

*Due to the dataset size being greater than GitHub's LFS max file size of 2GB, you can download [data.zip from Google Drive](https://drive.google.com/open?id=1zyt28sGXxvXaeSdj-kab2W_kWCH5fPIX), or [MEGA (has more versions of dataset in history)](https://mega.nz/#F!HewiSIzC!xJLv_0XkFLqkntdOMmFQAw).*

To start training, first unzip data.zip into the root directory of `traffic-lights` so that the directory tree looks like: `/traffic-lights/data/GREEN/etc.png`.

When you run train.py, it will then start a process of cropping images from the data set into the `/traffic-lights/data/flowed` directory.