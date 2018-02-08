'''
Created on 2018. 2. 5.

@author: hsw
'''

import os
import csv
import gzip
import collections
import re
import io

import xml.etree.ElementTree as ET

import requests
import pandas

#xml_path = os.path.join('download', 'full_database.zip')
#xml_path = os.path.join('download', 'drugbank_preview.xml.gz')
xml_path = os.path.join('download', 'drugbank.xml.gz')
with gzip.open(xml_path) as xml_file:
    tree = ET.parse(xml_file)
root = tree.getroot()


ns = '{http://www.drugbank.ca}'
inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"

rows = list()
for i, drug in enumerate(root):
    row = collections.OrderedDict()
    
    assert drug.tag == ns + 'drug'
    
    row['type'] = drug.get('type')
    
    row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
    
    row['name'] = drug.findtext(ns + "name")
    
    row['groups'] = [group.text for group in
        drug.findall("{ns}groups/{ns}group".format(ns = ns))]
    
    row['atc_codes'] = [code.get('code') for code in
        drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
    
    row['categories'] = [x.findtext(ns + 'category') for x in
        drug.findall("{ns}categories/{ns}category".format(ns = ns))]
    
    #row['product'] = [x.findtext(ns + 'product') for x in
    #    drug.findall("{ns}product/{ns}product".format(ns = ns))]
    
    
    #row['indication'] = [x.findtext(ns + 'indication') for x in
    #    drug.findall("{ns}indication/{ns}indication".format(ns = ns))]
    
    #row['description'] = [x.findtext(ns + 'description') for x in
    #    drug.findall("{ns}description/{ns}description".format(ns = ns))]
    
    #row['dosage-form'] = [x.findtext(ns + 'dosage-form') for x in
    #    drug.findall("{ns}dosage-form/{ns}dosage-form".format(ns = ns))]
    
    #row['route'] = [x.findtext(ns + 'route') for x in
    #    drug.findall("{ns}route/{ns}route".format(ns = ns))]
    
    
    
    row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
    row['inchikey'] = drug.findtext(inchikey_template.format(ns = ns))
    
    rows.append(row)



def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

rows = list(map(collapse_list_values, rows))


#columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'product', 'indication', 'description', 'dosage-form', 'route', 'inchikey', 'inchi']
columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'inchikey', 'inchi']
drugbank_df = pandas.DataFrame.from_dict(rows)[columns]
drugbank_df.head()

drugbank_slim_df = drugbank_df[
    drugbank_df.groups.map(lambda x: 'approved' in x) &
    drugbank_df.inchi.map(lambda x: x is not None) &
    drugbank_df.type.map(lambda x: x == 'small molecule')
]
drugbank_slim_df.head()



# write drugbank tsv(csv)
path = os.path.join('data', 'drugbank.csv')
drugbank_df.to_csv(path, sep=',', index=False)
#drugbank_df.to_csv(path, sep='\t', index=False)

# write slim drugbank tsv
#path = os.path.join('data', 'drugbank-slim.tsv')
#drugbank_slim_df.to_csv(path, sep='\t', index=False)


