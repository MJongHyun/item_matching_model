import errno


class mk_init():
    import os

    def __init__(self):
        # 한번에 전체 폴더 생성
        self.__path_lst = ['data/log/', 'data/base/']
        self.__test_path = ['data/result/']

    def mk_dir(self):
        try:
            for mk_path in self.__path_lst:
                if not (self.os.path.isdir(mk_path)):
                    self.os.makedirs(self.os.path.join(mk_path))
                    print("create directory")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise Exception("Failed to create directory")

    def mk_dir_dt(self, dt, sort):
        try:
            if sort != 'test':
                dt_path = []
                for path in self.__path_lst[1:]:
                    dt_path.append(path+dt)
                for mk_path in dt_path:
                    if not (self.os.path.isdir(mk_path)):
                        self.os.makedirs(self.os.path.join(mk_path))
                        print("create train directory")
            else:
                dt_path = []
                for path in self.__test_path:
                    dt_path.append(path + dt)
                for mk_path in dt_path:
                    if not (self.os.path.isdir(mk_path)):
                        self.os.makedirs(self.os.path.join(mk_path))
                        print("create test directory")

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise Exception("Failed to create directory")


