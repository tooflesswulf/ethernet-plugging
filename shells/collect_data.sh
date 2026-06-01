cd ./..
# for i in {29..31}
for i in 30 # 14 15 30
do
    sleep 5
    python teleoperation.py --id $i 
done
