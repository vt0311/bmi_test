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
    
    row['product'] = [x.findtext(ns + 'product') for x in
        drug.findall("{ns}product/{ns}product".format(ns = ns))]
    
    
    row['indication'] = [x.findtext(ns + 'indication') for x in
        drug.findall("{ns}indication/{ns}indication".format(ns = ns))]
    
    row['description'] = [x.findtext(ns + 'description') for x in
        drug.findall("{ns}description/{ns}description".format(ns = ns))]
    
    row['dosage-form'] = [x.findtext(ns + 'dosage-form') for x in
        drug.findall("{ns}dosage-form/{ns}dosage-form".format(ns = ns))]
    
    row['route'] = [x.findtext(ns + 'route') for x in
        drug.findall("{ns}route/{ns}route".format(ns = ns))]
    
    
    
    row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
    row['inchikey'] = drug.findtext(inchikey_template.format(ns = ns))
    
    rows.append(row)



def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

rows = list(map(collapse_list_values, rows))


columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'product', 'indication', 'description', 'dosage-form', 'route', 'inchikey', 'inchi']
#columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'product' 'indication', 'description', 'dosage-form', 'route', 'inchikey', 'inchi']
drugbank_df = pandas.DataFrame.from_dict(rows)[columns]
drugbank_df.head()

drugbank_slim_df = drugbank_df[
    drugbank_df.groups.map(lambda x: 'approved' in x) &
    drugbank_df.inchi.map(lambda x: x is not None) &
    drugbank_df.type.map(lambda x: x == 'small molecule')
]
drugbank_slim_df.head()



# write drugbank tsv
path = os.path.join('data', 'drugbank.csv')
drugbank_df.to_csv(path, sep=',', index=False)
#drugbank_df.to_csv(path, sep='\t', index=False)

# write slim drugbank tsv
path = os.path.join('data', 'drugbank-slim.tsv')
drugbank_slim_df.to_csv(path, sep='\t', index=False)


protein_rows = list()
for i, drug in enumerate(root):
    drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
    for category in ['target', 'enzyme', 'carrier', 'transporter']:
        proteins = drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category))
        for protein in proteins:
            row = {'drugbank_id': drugbank_id, 'category': category}
            row['organism'] = protein.findtext('{}organism'.format(ns))
            row['known_action'] = protein.findtext('{}known-action'.format(ns))
            actions = protein.findall('{ns}actions/{ns}action'.format(ns=ns))
            row['actions'] = '|'.join(action.text for action in actions)
            uniprot_ids = [polypep.text for polypep in protein.findall(
                "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier".format(ns=ns))]            
            if len(uniprot_ids) != 1: continue
            row['uniprot_id'] = uniprot_ids[0]
            ref_text = protein.findtext("{ns}references[@format='textile']".format(ns=ns))
            pmids = re.findall(r'pubmed/([0-9]+)', ref_text)
            row['pubmed_ids'] = '|'.join(pmids)
            protein_rows.append(row)

protein_df = pandas.DataFrame.from_dict(protein_rows)


# Read our uniprot to entrez_gene mapping
response = requests.get('http://git.dhimmel.com/uniprot/data/map/GeneID.tsv.gz', stream=True)
text = io.TextIOWrapper(gzip.GzipFile(fileobj=response.raw))
uniprot_df = pandas.read_table(text, engine='python')
uniprot_df.rename(columns={'uniprot': 'uniprot_id', 'GeneID': 'entrez_gene_id'}, inplace=True)

#print('Read our uniprot')


# merge uniprot mapping with protein_df
entrez_df = protein_df.merge(uniprot_df, how='inner')

#print('merge uniprot')


columns = ['drugbank_id', 'category', 'uniprot_id', 'entrez_gene_id', 'organism',
           'known_action', 'actions', 'pubmed_ids']
entrez_df = entrez_df[columns]

#print('to_csv')



path = os.path.join('data', 'proteins.tsv')
entrez_df.to_csv(path, sep='\t', index=False)

print('end')







