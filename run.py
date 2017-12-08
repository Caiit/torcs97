#! /usr/bin/env python3

# Start my-driver with port: -p 3001-3010
from pytocl.main import main
from my_driver import MyDriver

if __name__ == '__main__':
    main(MyDriver(logdata=False))
