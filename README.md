# traffic-lights

To download the full dataset, install [Resilio Sync](https://www.resilio.com/individuals/) and sync [this folder (27.6 GB).](https://link.resilio.com/#f=Traffic%20Lights%20Data&sz=3E10&t=2&s=I6YQPFIH6FCRURBIZPTET5JAB7X43U7QZHF6LWNFSCOAQDYQ3GGA&i=C426FTJRJUFVXKGK4WIR5U22BXEV44DFQ&v=2.6&a=2)

If you would like to download a sample of the dataset to see if it matches your needs, here's a few sample sets:
- [Small sample set, 200 images each class. Total zipped size: 707 MB](https://link.resilio.com/#f=Traffic%20Lights%20Small&sz=74E7&t=2&s=ZKDEP4CNVMHKYZCXE4ZZKYG4DO2UXG3RSWUPPCVNLB4JH4ZLFHJA&i=CEV5W25MHKRQQVP7OTRGS6RBH7T7NKZ3P&v=2.6&a=2)
- [Large sample set, 500 images each class. Total zipped size: 1.56 GB](https://link.resilio.com/#f=Traffic%20Lights%20Large&sz=16E8&t=2&s=KX5CB5LILKQ4STGPE2E6MY2RDYXVOCO3S42GIKZ6N5D5XW25UC6A&i=CFGKHUOSIEMJ2QL43W6HGM6BLFENSKQ37&v=2.6&a=2)

Each image is 1164x874.

To start training, first unzip one of the datasets into the root directory of `traffic-lights` so that the directory tree looks like: `/traffic-lights/data/GREEN/etc.png`.

When you run train.py, it will then start a process of cropping and randomly transforming images from the dataset into the `/traffic-lights/data/.processed` directory.

Depending on the amount of data and your CPU, you may want to decrease the amount of data generator threads [as defined here](train.py#L208) as it's pretty heavy on system resources. An ETA will print approximately every 15 seconds to give you feedback on how long it will take.
