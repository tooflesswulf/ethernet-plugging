DIR=../dataset/ethernet_plugin_unplug
for file in $(ls $DIR); do
    echo "python scripts/visualize-demo.py -f $DIR/$file/rawdata.h5";
    python scripts/visualize-demo.py -f $DIR/$file/rawdata.h5;
done
