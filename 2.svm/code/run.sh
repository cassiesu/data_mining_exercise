# cat ../data/xac | python mapper.py | python reducer.py  > weights.txt
# python evaluate.py weights.txt ../data/testdata.csv ../data/testlabel.csv .

cat ../data/xac | python mapper_wx.py | python reducer_wx.py  > weights.txt
python evaluate_wx.py weights.txt ../data/testdata.csv ../data/testlabel.csv .