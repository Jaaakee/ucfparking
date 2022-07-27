"""Get json results from API in order to cache them."""
import json

import requests


def cache_results(web_url):
    """Cache the results of the specified API."""
    request_list = [
        "/",
        "/all/",
        "/today/",
        "/week/",
        "/stats/",
        "/lastday/",
        "/lastmonth/",
        "/lastyear/",
    ]
    request_names = ["last", "all", "today", "week", "stats", "lastday", "lastmonth", "lastyear"]

    for index, request_to_perform in enumerate(request_list):
        content = requests.get(web_url + request_to_perform + "?cache=False")
        dict = json.loads(content.text)
        out_file = open(f"../content/{request_names[index]}.json", "w")
        json.dump(
            dict, out_file, ensure_ascii=False, allow_nan=False, indent=3, separators=(", ", ": ")
        )


if __name__ == "__main__":
    cache_results("https://api.ucfparking.com")
