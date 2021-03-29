import os, shutil
import fnmatch

def save_code_files(output_path, root_path):
    def match_patterns(include, exclude):
        def _ignore_patterns(path, names):
            # If current path in exclude list, ignore everything
            if path in set(name for pattern in exclude for name in fnmatch.filter([path], pattern)):
                return names
            # Get initial keep list from include patterns
            keep = set(name for pattern in include for name in fnmatch.filter(names, pattern))
            # Add subdirectories to keep list
            keep = set(list(keep) + [name for name in names if os.path.isdir(os.path.join(path, name))])
            # Remove exclude patterns from keep list
            keep_ex = set(name for pattern in exclude for name in fnmatch.filter(keep, pattern))
            keep = [name for name in keep if name not in keep_ex]
            # Ignore files not in keep list
            return set(name for name in names if name not in keep)

        return _ignore_patterns


    dst_dir = os.path.join(output_path, 'code')
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(root_path, dst_dir, ignore=match_patterns(include=['*.py', '*.yaml', '*.data', '*.cfg'],
                                                              exclude=['experiment*',
                                                                       '*.idea',
                                                                       '*__pycache__',
                                                                       'weights',
                                                                       'wandb',
                                                                       'asets'
                                                                       ]))
