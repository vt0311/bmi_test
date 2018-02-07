'''
Created on 2018. 2. 7.

@author: acorn
'''
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
from math import ceil

# http://info.childcare.go.kr/info/pnis/search/preview/AppraisaAuthenticationSlPu.jsp?flag=PI&STCODE_POP=11110000111
#[code1] - 프리뷰 들어가는 URL 만들기
imsi1 = []
imsi2 = []
result = []
def getUrl(numList):
    file = open ('c:/work/projectsave.csv', 'w', encoding='utf-8')

    for num in numList:
        base = "http://api.childcare.go.kr/mediate/rest/cpmsapi009/cpmsapi009/request?key=749a3ce41f5b497a92db7307f96c9e21&"
        parameters1 = "arcode=%s" % (num)

        url = base + parameters1
    
        html = urllib.request.urlopen( url )
        soup = BeautifulSoup(html, 'html.parser')
        
        
        kidsNametags = soup.find_all('stcode')#, attrs={'stcode':'name'}).text
#         print('유치원 이름 : ', kidsNametags)
        
        for tag in kidsNametags:
            a = tag.text
#             print(a)
            imsi1.append(tag.text)

        kidsNametags2 = soup.find_all('crname')#, attrs={'stcode':'name'}).text

        for tag2 in kidsNametags2:
            b = tag2.text
#             print(b)
            imsi2.append(tag2.text)
            
#         print(imsi1)
#         print(imsi2)
            print('진행중.')


    result1 = imsi1 +imsi2
    print(result1)
    file.write(result1)

    file.close()               


def main():
    
    numList = list(range(41110,41113))
    
    previewSearch = getUrl(numList)
     
    print(previewSearch)
     
    
if __name__ == '__main__':
    main()