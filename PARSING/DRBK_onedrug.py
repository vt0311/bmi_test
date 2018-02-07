fileDir = 'C:/Users/acorn/Desktop/'
fileName_read = 'full database.xml'

def checkN(data):
    print(data)

with open(fileDir+fileName_read, 'r', encoding='UTF-8') as rf:
        
    drug_flag = False
    temp = ''
    while True:
        rd = rf.readline()
        
        # end of file
        if(rd.strip() == ''):
            break
        
        if rd.startswith('<drug'):
            temp += rd
            drug_flag = True
        elif drug_flag and rd.startswith('</drug'):
            temp += rd
            drug_flag = False
            checkN(temp)
            temp = ''
        elif drug_flag:
            temp += rd
                
                
            