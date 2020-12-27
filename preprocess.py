# -*- coding: utf-8 -*-
# 제출 참고용 :  https://dacon.io/competitions/official/235401/codeshare/629?page=1&dtype=recent&ptype=pub

import csv

def read_csvFile(src):
    fr = open(src,'r',encoding='utf-8', errors="ignore")
    rdr = csv.reader(fr)

    data = [x for x in rdr]
    fr.close()
    return data

def write_tsvFile(src, data):
    fw = open(src,'w',encoding='utf-8')
    wrt = csv.writer(fw,delimiter='\t')

    for x in data:
        wrt.writerow(x)

    fw.close()


def preprocess_data(data):
    pass




if __name__ == '__main__':
    train_data, test_data = read_csvFile('./data/news_train.csv'), read_csvFile('./data/news_test.csv')
    # train_data = preprocess_data(train_data)
    # test_data = preprocess_data(test_data)

    write_tsvFile('./data/news_train.tsv', train_data)
    write_tsvFile('./data/news_test.tsv',test_data)


