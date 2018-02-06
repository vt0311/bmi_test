'''
Created on 2018. 2. 6.

@author: acorn
'''
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
from math import ceil
from pandas.core.common import union

result = []
def getUrl(numList):
    file = open ('c:/work/kids.csv', 'w', encoding='utf-8')
    for num in numList:
        base = "http://info.childcare.go.kr/info/pnis/search/preview/AppraisaAuthenticationSlPu.jsp?"
        parameters1 = "flag=PI&STCODE_POP=%s" % (num)
        parameters2 = "&CRNAMETITLE=%s" % (num)
        url = base + parameters1 + parameters2
    
        html = urllib.request.urlopen( url )
        soup = BeautifulSoup(html, 'html.parser')
    
        kidsNametags = soup.find('span', attrs={'class':'name'}).text
        print('유치원 이름 : ', kidsNametags)
    
        tags = soup.find('tbody').text
#         print(tags)

        for tag in tags:
            v = tags.replace('종합평가서', '')
            v2 = v.replace('종합평가서 보기', '')
            v3 = v2.replace('\n', '')
            v4 = v3.replace('보기', '')
            v5 = v4.replace('총점', '')
            v6 = v5.replace('보육환경', '')
            v7 = v6.replace('운영관리', '')
            v8 = v7.replace('보육과정', '')
            v9 = v8.replace('상호작용과 교수법', '')
            v10 = v9.replace('건강과 영양', '')
            v11 = v10.replace('안전', '')
            v12 = v11.replace('및', '')
            v13 = v12.replace('평가인증여부인증지표구분3차 지표', '')
#             print(v13)

        print('평가점수 : ', v13)
        
        imsi1 = ''
        imsi2 = []
        result = []
        result2 = []
        for item in v13:
            if item == '-':
                imsi1 = item
                result.append(imsi1)
            else:
#                 print(a)    
                
                
 #       print(result)
#       print(result2)    
        
#        result2 = str(result2)
#         result3 = float(result2)
#         print(result3)
        
        
        
#         file.write(total)
    
 #   file.close()    
 #   return (v13,kidsNametags)    


def main():
    jsonResult = [] # 결과를 저장할 리스트
    
    numList = list(range(11110000070,11110000150))
    
    previewSearch = getUrl(numList)
     
    print('aaa:', previewSearch)
     
    
if __name__ == '__main__':
    main()