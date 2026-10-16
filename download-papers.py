# Original script from reddit https://www.reddit.com/r/algotrading/comments/g2gvrn/500_recent_research_papers_on_algorithmic_trading/
# No longer functional but provided the link which stays current

import urllib.request as urllib2

import requests
from bs4 import BeautifulSoup

pattern = "http://arxiv.org/abs/"
url = "https://resources.paperdigest.org/2020/04/recent-papers-on-algorithmic-trading-high-frequency-trading/"
illegal = ['<>:"/\\|?*']

req = requests.get(url)
content = req.content
soup = BeautifulSoup(content, "html.parser")


table = soup.find_all("table")[0]
targets = {}

rows = table.find_all("tr")[1:]  # Skip header row

for row in rows:
    cells = row.find_all("td")
    if len(cells) >= 2:
        paper_cell = cells[1]  # Paper title is in the second column
        arxiv_link = paper_cell.find("a", href=lambda x: x and "arxiv.org/abs/" in x)

        if arxiv_link:
            # Extract title before "Related Papers"
            full_text = paper_cell.get_text()
            title = full_text.split("Related Papers")[0].strip()

            # Remove illegal filename characters
            for char in '<>:"/\\|?*':
                title = title.replace(char, "")

            targets[arxiv_link["href"]] = title

print(f"Found {len(targets)} papers")
links = []
title = []
for link in targets:
    links.append(link)
    title.append(targets[link])

for i in range(len(links)):
    link = links[i].replace("/abs/", "/pdf/")
    if link.startswith("http:"):
        link = link.replace("http:", "https:")

    filename = title[i][:100] + ".pdf"  # Limit filename length

    try:
        open_url = urllib2.urlopen(link)
        with open(filename, "wb") as file:
            file.write(open_url.read())
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Could not get: {title[i][:50]}... - {e!s}")
