from xml.etree.ElementTree import parse


fileDir = ''
fileName_read = 'parsed1.xml'
fileName_write = 'parsed1_N.xml'
fileName_writeNone = 'parsed1_None.xml'

def checkN(data):
    t = '<?xml version="1.0" encoding="UTF-8"?>\n'
    data = t+data
    with open(fileDir+'temp.xml', 'w', encoding='UTF-8') as tf:
        tf.write(data)
    
    tr = parse(fileDir+'temp.xml').getroot()
    atcCodes = tr.find('atc-codes').findall('atc-code')
    
    if len(atcCodes) == 0: # No atc-code
        return 0
    
    for atcCode in atcCodes:
        if atcCode.get('code').startswith('N'):
            return 1
    return -1


with open(fileDir+fileName_read, 'r', encoding='UTF-8') as rf:
    with open(fileDir+fileName_write, 'w', encoding='UTF-8') as wf:
        with open(fileDir+fileName_writeNone, 'w', encoding='UTF-8') as wnf:
            wf.write(rf.readline()) # the first tag <- <drugbank>
            
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
                    
                    checked = checkN(temp)
                    if checked == 1:
                        wf.write(temp)
                    elif checked == 0:
                        wnf.write(temp)
                    temp = ''
                    
                elif drug_flag:
                    temp += rd
                
