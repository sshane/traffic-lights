# traffic-lights

*Due to the dataset size being greater than GitHub's LFS max file size of 2GB, you can download [data.zip from Google Drive](https://drive.google.com/open?id=1SGdPQmeVqC_uhzOHk2oEm7DzAN9NeCsA), or [MEGA (has more versions of dataset in history)](https://mega.nz/#F!HewiSIzC!xJLv_0XkFLqkntdOMmFQAw).*
If you'd like to use Resilio Sync to download the dataset, here's our [dataset folder link](https://link.resilio.com/#f=Traffic%20Light%20Data&sz=11E9&t=2&s=TJ7UO2MWE5J3FPMNYUEO4UYAA65B7PKE4D7BSQV7GHTZGJC6GAAA&i=CPOVRRCNKD6SVK3DML33A3GJPZHAPZDGI&v=2.6&a=2). Click, and it will start syncing from my server desktop.

To start training, first unzip data.zip into the root directory of `traffic-lights` so that the directory tree looks like: `/traffic-lights/data/GREEN/etc.png`.

When you run train.py, it will then start a process of cropping images from the data set into the `/traffic-lights/data/.processed` directory.