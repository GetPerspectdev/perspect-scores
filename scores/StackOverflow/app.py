#Collect Stack overflow data

import requests
from bs4 import BeautifulSoup as bs
from bs4 import Tag
import json
import re

def get_stack_data(url):
    r = requests.get(url, timeout=10)

    soup = bs(r.content, "html.parser")

    ###GET STATS
    stats = soup.find("div", {"id": "stats"})

    stat_divs = stats.find("div", {"class": "d-flex flex__allitems6 gs16 fw-wrap md:jc-space-between"})

    stat_list = []

    for i, inner_divs in enumerate(stat_divs):
        # This break is necessary for 99% of Stack Overflow users who don't have "Top x% of users" in their profile stats
        if i == 9:
            break
        for div in inner_divs:
            if isinstance(div, Tag):
                stat = div.get_text().strip().replace(",","")
                if 'm' in stat:
                    stat = stat.replace("m","")
                    if '.' in stat:
                        stat = stat.replace(".","")
                        stat = int(stat)
                        stat = stat * 100000
                    else:
                        stat = stat * 1000000
                else:
                    val_to_append = int(stat)
                stat_list.append(val_to_append)

    stats = {"reputation": stat_list[0],
            "reached": stat_list[1],
            "answers": stat_list[2],
            "questions": stat_list[3]}

    stats = json.dumps(stats)

    ###GET BADGES
    badge_block = soup.find("div", {"class": "d-flex flex__fl-equal fw-wrap gs24"})
    badge_divs = soup.findAll(class_= 'd-flex fd-column jc-space-between h100 g12')

    badge_list = []
    for div in badge_divs:
        if div.find_all("div", {"class": "d-flex ai-center"}):
            badge = div.find("div", {"class": "fs-title fw-bold fc-black-800"})
            badge_num = int(badge.get_text().strip())
            val_to_append = badge_num
        else:
            val_to_append = 0
        
        badge_list.append(val_to_append)

    badges = {"gold": badge_list[0],
            "silver": badge_list[1],
            "bronze": badge_list[2]}

    badges = json.dumps(badges)

    print(stats, badges)

    return stats, badges

url = "https://stackoverflow.com/users/5459839/trincot"

url += "?tab=profile"

get_stack_data(url)
