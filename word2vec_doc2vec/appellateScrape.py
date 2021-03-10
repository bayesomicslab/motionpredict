from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import requests
import pandas as pd
import os
from tika import parser
import sys

#outputs a txt file called AppellateOpinionLegalData.txt with all of the dowloaded pdf court opinion documents
#to run: python3 appellateScrape.py

class ScrapingDataToTxt():
    def __init__(self, directory):
        self.directory = directory

    def pdfToTxt(self):
        #loops through the pdfs in the directory and appends them to a txt file
        print("looping through files...")
        count = 0
        for file in os.listdir(self.directory):
            if file.endswith(".pdf"):
                print("opening files...")
                pdfFileObj = open(os.path.join(self.directory, file), "rb")
                print("parsering...")
                rawText = parser.from_file(pdfFileObj)
                print(f"writing to txt file {count}....")
                count += 1
                Data = rawText['content']
                try:
                    strippedData = Data.strip()
                    formatData = " ".join(strippedData.split())
                    f = open("AppellateOpinionLegalData.txt", "a")
                    f.write(f"{count}\t{formatData}\n")
                    f.close()
                except AttributeError:
                    continue

    def PdfDownload(self):
        #access the documents through url and downloads the content to a local directory called LawPdfFilesScraped
        print("creating list of urls....")
        urls = []
        pages = []
        for i in range(21):
            pages.append("%02d" % i)
        for n in pages:
            urls.append(f"https://jud.ct.gov/external/supapp/archiveAROap{n}.htm")

        folder_location = "{}".format(self.directory)
        os.mkdir(folder_location)
        print("accessing url....")
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            print("downloading files from website....")
            for link in soup.select("a[href$='.pdf']"):
                filename = os.path.join(folder_location,link["href"].split('/')[-1])
                with open(filename, "wb") as f:
                    f.write(requests.get(urljoin(url,link["href"])).content)

        self.pdfToTxt()



if __name__ == '__main__':
    prepData = ScrapingDataToTxt("./LawPdfFilesScraped")
    prepData.PdfDownload()
