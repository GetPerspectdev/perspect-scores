import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

basic = ('chrisbrousseau304@gmail.com', )
verbose = True
repo_url = 'https://github.com/GetPerspectdev/DesignPatternDetector'
url_tree = [repo_url]
file_urls = []
for i in url_tree:
        r = requests.get(i, auth=basic)
        if r.status_code == 200:
            if verbose:
                print(f"Creating url tree: {i}")
            soup = BeautifulSoup(r.content, 'html.parser')
            if len(soup.find_all('a')) > 0:
                for j in soup.find_all('a'):
                        try:
                            if "tree/main" in j.get('href') or "tree/main" in j.get('href'):
                                url = urljoin(i, j.get('href'))
                                if url not in url_tree:
                                    url_tree.append(url)
                        except TypeError:
                            pass
            else:
              stuff = json.loads(str(soup))
              try:
                for j in stuff['payload']['tree']['items']:
                    try:
                        if f"{i.split('/')[-1]}" in j['path'] and len(j['path'].split(".")) == 1:
                            url = f"{i}/{j['name']}"
                            if url not in url_tree:
                                    url_tree.append(url)
                    except TypeError:
                            print(j)
                print(stuff['payload']['tree']['items'])
              except KeyError:
                   print(stuff['payload'].items())
for tree_url in url_tree:
        if verbose:
                print(f"Tree: {tree_url}")
        r = requests.get(tree_url, auth=basic)
        soup = BeautifulSoup(r.content, 'html.parser')
        if len(soup.find_all('a')) > 0:
            for i in soup.find_all('a'):
                try:
                    if f"blob/main" in i.get('href'):
                        url = urljoin(repo_url, i.get('href'))
                        if url not in file_urls:
                            file_urls.append(url)
                            if verbose:
                                print(f"File: {url.split('/')[-1]}")
                except TypeError:
                    if verbose:
                        print(f"Error on {i}")
        else:
            stuff = json.loads(str(soup))
            try:
                for j in stuff['payload']['tree']['items']:
                    try:
                        if j['contentType'] == 'file':
                            url = f"{tree_url}/{j['name']}"
                            if url not in file_urls:
                                    file_urls.append(url)
                                    if verbose:
                                        print(f"File: {url.split('/')[-1]}")
                    except TypeError:
                        if verbose:
                            print(f"Error on {j}")
            except KeyError:
                if verbose:
                    print(f"Error on {i}")
print(url_tree)
print(file_urls)
            # for j in soup.find_all('body'):
            #         print(j)
            #         try:
            #             stuff = json.load(j)
            #             if f"{repo_url.split('/')[-1]}" in stuff.keys():
            #                 url = urljoin(repo_url, stuff[f"{repo_url.split('/')[-1]}"])
            #                 if url not in url_tree:
            #                     url_tree.append(url)
            #                     print(f"Found New Folder: {url_tree[-1]}")
            