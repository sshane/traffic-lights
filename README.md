# traffic-lights

*Due to the dataset size being greater than GitHub's LFS max file size of 2GB, you can download [our dataset from Google Drive](https://drive.google.com/open?id=1SGdPQmeVqC_uhzOHk2oEm7DzAN9NeCsA), or [MEGA](https://mega.nz/#F!aDhARarQ!vfos_p1yLbj9BvJG69zsQw).*

To start training, first unzip data.zip into the root directory of `traffic-lights` so that the directory tree looks like: `/traffic-lights/data/GREEN/etc.png`.

When you run train.py, it will then start a process of cropping and randomly transforming images from the dataset into the `/traffic-lights/data/.processed` directory.

Depending on the amount of data and your CPU, you may want to decrease the amount of data generator threads as [defined here](train.py#L202) as it's pretty heavy on system resources. An ETA will print approximately every 15 seconds to give you feedback on how long it will take.