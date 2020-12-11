# Authors:
# Brandon Xue       brandonx@csu.fullerton.edu
# Jacob Rapmund     jacobwrap86@csu.fullerton.edu
#
# This module contains some basic utilities.

def has_extension(filepath: str, ext: str) -> bool:
    ''' Check if the given file path has the given extension. '''
    return filepath.rfind(ext) == (len(filepath) - len(ext))

def has_trailing_slash(dirpath: str) -> str:
    if len(dirpath) == 0:
        return False
    return dirpath.rfind('/') == (len(dirpath) - 1)

def add_trailing_slash(dirpath: str) -> str:
    ''' Add a trailing forward slash to a directory path if not exists .'''
    if not has_trailing_slash(dirpath):
        dirpath = dirpath + '/'
    return dirpath

def find_parent_child(dirpath: str, slash=True) -> str:
    ''' Given a subdirectory's path, find the parent directory's path. '''
    if len(dirpath) == 0: # Check if empty str
        return ""
    search_path = dirpath # Get rid of trailing slash for search
    if dirpath[-1] == '/':
        search_path = dirpath[:-1]
    i = search_path.rfind('/')
    if slash:
        parent, child = search_path[:i+1], search_path[i+1:]
    else:
        parent, child = search_path[:i], search_path[i+1:]
    return parent, child