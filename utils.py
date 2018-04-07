import os
def get_dir_filelist_by_extension(dir, ext,with_parent_path=False):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.split('.')[-1] == ext:
            if with_parent_path:
                file_list.append(dir + '/' + item)
            else:
                file_list.append(item)
    return file_list