for file in $(ls ../dataset/ethernet_plug_v3/); do
    echo "python scripts/visualize-demo.py -f ../dataset/ethernet_plug_v3/$file/rawdata.h5";
    python scripts/visualize-demo.py -f ../dataset/ethernet_plug_v3/$file/rawdata.h5;
done

