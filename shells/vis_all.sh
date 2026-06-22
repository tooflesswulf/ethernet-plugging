for file in $(ls logs_pluginv2_local_delta_50); do
    echo "python scripts/visualize-demo.py -f logs_pluginv2_local_delta_50/$file/rawdata.h5";
    python scripts/visualize-demo.py -f logs_pluginv2_local_delta_50/$file/rawdata.h5;
done

