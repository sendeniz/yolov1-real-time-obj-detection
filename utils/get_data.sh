#!/bin/bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
rm VOCtrainval_06-Nov-2007.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

wget https://pjreddie.com/media/files/voc_label.py
python3 utils/voc_label.py

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
mv test.txt old_txt_files/
mv train.txt old_txt_files/