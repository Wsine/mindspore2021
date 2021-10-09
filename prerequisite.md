```bash
sudo apt install m4
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xf gmp/gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --prefix=/path/to/mindspore2021/vendor/gmp-6.1.2 --enable-cxx
make -j4
make install
```
