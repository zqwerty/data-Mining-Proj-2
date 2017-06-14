import os
import cPickle as pickle
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def test():
    tree = ET.ElementTree(file='../data_slice/1801293.xml')
    root = tree.getroot()
    head = root.find('head')
    body = root.find('body')
    title = head.find('title').text
    doc_id = head.find('docdata').find('doc-id').attrib['id-string']
    locations = []
    for c in tree.iter('location'):
        locations.append(c.text)
    for c in head.findall('meta'):
        if 'publication_day_of_month' == c.attrib['name']:
            print c.attrib['content']
        if 'publication_month' == c.attrib['name']:
            print c.attrib['content']
        if 'publication_year' == c.attrib['name']:
            print c.attrib['content']
        if 'online_sections' == c.attrib['name']:
            print c.attrib['content']
    full_text = ""
    for c in tree.iter('block'):
        if 'full_text' == c.attrib['class']:
            for c2 in c.findall('p'):
                full_text += c2.text
    print full_text


def readXML(folder='../data_slice'):
    '''

    :param folder: xml dir
    :return: docs:[(doc-id,category,full_text),...]
    '''
    for root, dirs, files in os.walk(folder):
        fs = [os.path.join(root, x) for x in files if 'xml' in x]

    # extract doc-id, category, full-text
    docs = []
    for f in fs:
        tree = ET.parse(f)
        # doc-id
        for c in tree.iter('doc-id'):
            docid = c.attrib['id-string']
        print docid

        # category -list
        for c in tree.find('head').findall('meta'):
            if 'online_sections' == c.attrib['name']:
                category = c.attrib['content'].split('; ')

        # full-text
        full_text = ''
        for c in tree.iter('block'):
            if 'full_text' == c.attrib['class']:
                for c2 in c.findall('p'):
                    full_text += c2.text
        try:
            docs.append((docid, category, full_text))
        except ReferenceError:
            print 'error! %s:%s' % (docid, category)
    return docs

if __name__ == '__main__':
    ds = readXML()
    pickle.dump(ds,open('../data_slice/data.p','wb'))
    dds = pickle.load(open('../data_slice/data.p'))
    print len(dds)
    print dds[:10]