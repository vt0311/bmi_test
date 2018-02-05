'''
Created on 2018. 2. 5.

@author: acorn
'''



from skbio import DNA, RNA

d = DNA('ACCGGGTA')
d = DNA('ACCGGGTA', metadata={'id':"my-sequence", 'description':"GFP"},
          positional_metadata={'quality':[22, 25, 22, 18, 23, 25, 25, 25]})
d = DNA('ACCGGTA', metadata={'id':"my-sequence"})

d1 = DNA('.ACC--GGG-TA...', metadata={'id':'my-sequence'})
d2 = d1.degap()
print(d2)

d3 = d2.reverse_complement()
print(d3)

