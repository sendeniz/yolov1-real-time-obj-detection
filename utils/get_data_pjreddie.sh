#!/bin/bash
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar 
tar xf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar

wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar 
tar xf VOCtrainval_06-Nov-2007.tar
rm VOCtrainval_06-Nov-2007.tar

wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

#wget https://pjreddie.com/media/files/voc_label.py
#python3 utils/voc_label.py

cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt
mkdir old_txt_files
mv 2007* 2012* old_txt_files/

mkdir data
mkdir data/images
mkdir data/labels

mv VOCdevkit/VOC2007/JPEGImages/*.jpg data/images/                                      
mv VOCdevkit/VOC2012/JPEGImages/*.jpg data/images/                                      
mv VOCdevkit/VOC2007/labels/*.txt data/labels/                                          
mv VOCdevkit/VOC2012/labels/*.txt data/labels/

rm -rf VOCdevkit/
python3 utils/generate_csv.py
mv test.txt old_txt_files/
mv train.txt old_txt_files/
mv test.csv data/
mv train.csv data/
rm -r old_txt_files/