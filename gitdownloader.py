# Based on https://github.com/sdushantha/gitdir

import re
import os
import urllib.request
import json


class Downloader:
    def __init__(self, url):
        self.repo_url = url
        self.output_dir = './'
        re_branch = re.compile("/(tree|blob)/(.+?)/")
        branch = re_branch.search(self.repo_url)
        self.download_dirs = self.repo_url[branch.end():]
        self.api_url = (self.repo_url[:branch.start()].replace("github.com", "api.github.com/repos", 1) +
                        "/contents/" + self.download_dirs + "?ref=" + branch.group(2))

    def download(self):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            response = urllib.request.urlretrieve(self.api_url)
            os.makedirs(self.output_dir, exist_ok=True)
        except:
            return

        total_files = 0

        with open(response[0], "r") as f:
            data = json.load(f)
            total_files += len(data)

            if isinstance(data, dict) and data["type"] == "file":
                try:
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(data["download_url"], os.path.join(self.output_dir, data["name"]))

                    return total_files

                except:
                    return

            for file in data:
                file_url = file["download_url"]

                path = file["path"]
                dirname = os.path.dirname(path)

                if dirname != '':
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                else:
                    pass

                if file_url is not None:
                    try:
                        opener = urllib.request.build_opener()
                        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                        urllib.request.install_opener(opener)
                        urllib.request.urlretrieve(file_url, path)
                    except:
                        return

        return total_files
